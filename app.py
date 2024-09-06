import gradio as gr
import pandas as pd
import logging
import asyncio
import os
import re
import json
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from huggingface_hub import CommitScheduler
from auditqa.sample_questions import QUESTIONS
from auditqa.reports import files, report_list
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import HuggingFaceEndpoint
from auditqa.process_chunks import load_chunks, getconfig
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from qdrant_client.http import models as rest
from dotenv import load_dotenv
load_dotenv()
# token to allow acces to Hub, This token should also be 
# valid fo calls made to Inference endpoints
HF_token = os.environ["HF_TOKEN"]

# create the local logs repo
JSON_DATASET_DIR = Path("json_dataset")
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)
JSON_DATASET_PATH = JSON_DATASET_DIR / f"logs-{uuid4()}.json"

# the logs are written to dataset repo
# https://huggingface.co/spaces/Wauplin/space_to_dataset_saver
scheduler = CommitScheduler(
    repo_id="GIZ/spaces_logs",
    repo_type="dataset",
    folder_path=JSON_DATASET_DIR,
    path_in_repo="audit_chatbot",
)

model_config = getconfig("model_params.cfg")



#### VECTOR STORE ####
# reports contain the already created chunks from Markdown version of pdf reports
# document processing was done using : https://github.com/axa-group/Parsr
vectorstores = load_chunks()


#### FUNCTIONS ####
# App UI and and its functionality is inspired and adapted from
# https://huggingface.co/spaces/Ekimetrics/climate-question-answering


def save_logs(logs) -> None:
    """ Every interaction with app saves the log of question and answer, 
        this is to get the usage statistics of app and evaluate model performances 
    """
    with scheduler.lock:
        with JSON_DATASET_PATH.open("a") as f:
            json.dump(logs, f)
            f.write("\n")
    logging.info("logging done")


def make_html_source(source,i):
    """
    takes the text and converts it into html format for display in "source" side tab
    """
    meta = source.metadata
    content = source.page_content.strip()

    name = meta['filename']
    card = f"""
        <div class="card" id="doc{i}">
            <div class="card-content">
                <h2>Doc {i} - {meta['filename']} - Page {int(meta['page'])}</h2>
                <p>{content}</p>
            </div>
            <div class="card-footer">
                <span>{name}</span>
                <a href="{meta['filename']}#page={int(meta['page'])}" target="_blank" class="pdf-link">
                    <span role="img" aria-label="Open PDF">🔗</span>
                </a>
            </div>
        </div>
        """

    return card

def parse_output_llm_with_sources(output):
    # Split the content into a list of text and "[Doc X]" references
    content_parts = re.split(r'\[(Doc\s?\d+(?:,\s?Doc\s?\d+)*)\]', output)
    parts = []
    for part in content_parts:
        if part.startswith("Doc"):
            subparts = part.split(",")
            subparts = [subpart.lower().replace("doc","").strip() for subpart in subparts]
            subparts = [f"""<a href="#doc{subpart}" class="a-doc-ref" target="_self"><span class='doc-ref'><sup>{subpart}</sup></span></a>""" for subpart in subparts]
            parts.append("".join(subparts))
        else:
            parts.append(part)
    content_parts = "".join(parts)
    return content_parts

def start_chat(query,history):
    history = history + [(query,None)]
    history = [tuple(x) for x in history]
    return (gr.update(interactive = False),gr.update(selected=1),history)

def finish_chat():
    return (gr.update(interactive = True,value = ""))
    
async def chat(query,history,sources,reports,subtype,year):
    """taking a query and a message history, use a pipeline (reformulation, retriever, answering) 
       to yield a tuple of:(messages in gradio format/messages in langchain format, source documents)
    """

    logging.info(f">> NEW QUESTION : {query}")
    logging.info(f"history:{history}")
    #print(f"audience:{audience}")
    logging.info(f"sources:{sources}")
    logging.info(f"reports:{reports}")
    logging.info(f"subtype:{subtype}")
    logging.info(f"year:{year}")
    docs_html = ""
    output_query = ""

    ##------------------------fetch collection from vectorstore------------------------------
    vectorstore = vectorstores["allreports"]
    ##---------------------construct filter for metdata filtering---------------------------
    if len(reports) == 0:
        ("defining filter for:{}:{}:{}".format(sources,subtype,year))
        filter=rest.Filter(
                must=[rest.FieldCondition(
                        key="metadata.source",
                        match=rest.MatchValue(value=sources)
                    ),
                    rest.FieldCondition(
                        key="metadata.subtype",
                        match=rest.MatchValue(value=subtype)
                    ),
                    rest.FieldCondition(
                        key="metadata.year",
                        match=rest.MatchAny(any=year)
                    ),])
    else:
        print("defining filter for allreports:",reports)
        filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.filename",
                        match=rest.MatchAny(any=reports)
                    )])
        

    ##------------------------------get context---------------------------------------------- 
    context_retrieved_lst = []
    question_lst= [query]

    for question in question_lst:
        # similarity score threshold can be used to make adjustments in quality and quantity for Retriever
        # However need to make balancing, as retrieved results are again used by Ranker to fetch best among
        # retreived results
        retriever = vectorstore.as_retriever(
          search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6, 
                                                                   "k": int(model_config.get('retriever','TOP_K')), 
                                                                   "filter":filter})
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        context_retrieved = compression_retriever.invoke(question)
        logging.info(len(context_retrieved))
        for doc in context_retrieved:
            logging.info(doc.metadata)
    
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
    
        context_retrieved_formatted = format_docs(context_retrieved)
        context_retrieved_lst.append(context_retrieved_formatted)

    ##------------------- -------------Prompt--------------------------------------------------
    SYSTEM_PROMPT = """
        You are AuditQ&A, an AI Assistant created by Auditors and Data Scientist. You are given a question and extracted passages of the consolidated/departmental/thematic focus audit reports. Provide a clear and structured answer based on the passages/context provided and the guidelines.
        Guidelines:
        - If the passages have useful facts or numbers, use them in your answer.
        - When you use information from a passage, mention where it came from by using [Doc i] at the end of the sentence. i stands for the number of the document.
        - Do not use the sentence 'Doc i says ...' to say where information came from.
        - If the same thing is said in more than one document, you can mention all of them like this: [Doc i, Doc j, Doc k]
        - Do not just summarize each passage one by one. Group your summaries to highlight the key parts in the explanation.
        - If it makes sense, use bullet points and lists to make your answers easier to understand.
        - You do not need to use every passage. Only use the ones that help answer the question.
        - If the documents do not have the information needed to answer the question, just say you do not have enough information.
        """
    
    USER_PROMPT = """Passages:
        {context}
        -----------------------
        Question: {question}  - Explained to audit expert
        Answer in english with the passages citations:
        """.format(context = context_retrieved_lst, question=query)

    messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=USER_PROMPT
                ),]

    ##-----------------------getting inference endpoints------------------------------

    # Set up the streaming callback handler
    callback = StreamingStdOutCallbackHandler()

    # Initialize the HuggingFaceEndpoint with streaming enabled
    llm_qa = HuggingFaceEndpoint(
        endpoint_url=model_config.get('reader', 'ENDPOINT'),
        max_new_tokens=512,
        repetition_penalty=1.03,
        timeout=70,
        huggingfacehub_api_token=HF_token,
        streaming=True, # Enable streaming for real-time token generation
        callbacks=[callback] # Add the streaming callback handler
    )

    # Create a ChatHuggingFace instance with the streaming-enabled endpoint
    chat_model = ChatHuggingFace(llm=llm_qa)

    # Prepare the HTML for displaying source documents
    docs_html = []
    for i, d in enumerate(context_retrieved, 1):
        docs_html.append(make_html_source(d, i))
    docs_html = "".join(docs_html)

    # Initialize the variable to store the accumulated answer
    answer_yet = ""

    # Define an asynchronous generator function to process the streaming response
    async def process_stream():
        # Without nonlocal, Python would create a new local variable answer_yet inside process_stream(), instead of modifying the one from the outer scope.
        nonlocal answer_yet # Use the outer scope's answer_yet variable
        # Iterate over the streaming response chunks
        async for chunk in chat_model.astream(messages):
            token = chunk.content
            answer_yet += token
            parsed_answer = parse_output_llm_with_sources(answer_yet)
            history[-1] = (query, parsed_answer)
            yield [tuple(x) for x in history], docs_html

    # Stream the response updates
    async for update in process_stream():
        yield update

    # #callbacks = [StreamingStdOutCallbackHandler()]
    # llm_qa = HuggingFaceEndpoint(
    #     endpoint_url= model_config.get('reader','ENDPOINT'),
    #     max_new_tokens=512,
    #     repetition_penalty=1.03,
    #     timeout=70,
    #     huggingfacehub_api_token=HF_token,)

    # # create RAG
    # chat_model = ChatHuggingFace(llm=llm_qa)
    
    # ##-------------------------- get answers ---------------------------------------
    # answer_lst = []
    # for question, context in zip(question_lst , context_retrieved_lst):
    #     answer = chat_model.invoke(messages)
    #     answer_lst.append(answer.content)
    # docs_html = []
    # for i, d in enumerate(context_retrieved, 1):
    #     docs_html.append(make_html_source(d, i))
    # docs_html = "".join(docs_html)

    # previous_answer = history[-1][1]
    # previous_answer = previous_answer if previous_answer is not None else ""
    # answer_yet = previous_answer + answer_lst[0]
    # answer_yet = parse_output_llm_with_sources(answer_yet)
    # history[-1] = (query,answer_yet)
    
    # history = [tuple(x) for x in history]
        
    # yield history,docs_html

    # logging the event
    try:
        timestamp = str(datetime.now().timestamp())
        logs = {
                "system_prompt": SYSTEM_PROMPT,
                "sources":sources,
                "reports":reports,
                "subtype":subtype,
                "year":year,
                "question":query,
                "sources":sources,
                "retriever":model_config.get('retriever','MODEL'),
                "raeder":model_config.get('reader','MODEL'),
                "docs":[doc.page_content for doc in context_retrieved],
                "answer": history[-1][1],
                "time": timestamp,
            }
        save_logs(logs)
    except Exception as e:
        logging.error(e)




#### Gradio App ####

# Set up Gradio Theme
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="red",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
    text_size = gr.themes.utils.sizes.text_sm,
)

init_prompt =  """
Hello, I am Audit Q&A, a conversational assistant designed to help you understand audit Reports. I will answer your questions by using **Audit reports publishsed by Auditor General Office**.
💡 How to use (tabs on right)
- **Reports**: You can choose to address your question to either specific report or a collection of report like District or Ministry focused reports. \
If you dont select any then the Consolidated report is relied upon to answer your question.
- **Examples**: We have curated some example questions,select a particular question from category of questions.
- **Sources**: This tab will display the relied upon paragraphs from the report, to help you in assessing or fact checking if the answer provided by Audit Q&A assitant is correct or not.
⚠️ For limitations of the tool please check **Disclaimer** tab.
"""


with gr.Blocks(title="Audit Q&A", css= "style.css", theme=theme,elem_id = "main-component") as demo:
    #----------------------------------------------------------------------------------------------
    # main tab where chat interaction happens
    # ---------------------------------------------------------------------------------------------
    with gr.Tab("AuditQ&A"):
        
        with gr.Row(elem_id="chatbot-row"):
            # chatbot output screen
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    value=[(None,init_prompt)],
                    show_copy_button=True,show_label = False,elem_id="chatbot",layout = "panel",
                    avatar_images = (None,"data-collection.png"),
                )
                



                with gr.Row(elem_id = "input-message"):
                    textbox=gr.Textbox(placeholder="Ask me anything here!",show_label=False,scale=7,
                                       lines = 1,interactive = True,elem_id="input-textbox")

            # second column with playground area for user to select values
            with gr.Column(scale=1, variant="panel",elem_id = "right-panel"):
                # creating tabs on right panel
                with gr.Tabs() as tabs:
                    #---------------- tab for REPORTS SELECTION ----------------------
                    
                    with gr.Tab("Reports",elem_id = "tab-config",id = 2):
                        gr.Markdown("Reminder: To get better results select the specific report/reports")

                        
                        #----- First level filter for selecting Report source/category ----------
                        dropdown_sources = gr.Radio(
                            ["Consolidated", "District","Ministry"],
                            label="Select Report Category",
                            value="Consolidated",
                            interactive=True,
                        )

                        #------ second level filter for selecting subtype within the report category selected above
                        dropdown_category = gr.Dropdown(
                            list(files["Consolidated"].keys()),
                            value = list(files["Consolidated"].keys())[0],
                            label = "Filter for Sub-Type",
                            interactive=True)

                        #----------- update the secodn level filter abse don values from first level ----------------
                        def rs_change(rs):
                            return gr.update(choices=files[rs], value=list(files[rs].keys())[0])
                        dropdown_sources.change(fn=rs_change, inputs=[dropdown_sources], outputs=[dropdown_category])

                        #--------- Select the years for reports -------------------------------------
                        dropdown_year = gr.Dropdown(
                            ['2018','2019','2020','2021','2022'],
                            label="Filter for year",
                            multiselect=True,
                            value=['2022'],
                            interactive=True,
                        )
                        gr.Markdown("-------------------------------------------------------------------------")
                        #---------------- Another way to select reports across category and sub-type ------------
                        dropdown_reports = gr.Dropdown(
                        report_list,
                        label="Or select specific reports",
                        multiselect=True,
                        value=[],
                        interactive=True,)

                    ############### tab for Question selection ###############
                    with gr.TabItem("Examples",elem_id = "tab-examples",id = 0):
                        examples_hidden = gr.Textbox(visible = False)

                        # getting defualt key value to display
                        first_key = list(QUESTIONS.keys())[0]
                        # create the question category dropdown
                        dropdown_samples = gr.Dropdown(QUESTIONS.keys(),value = first_key,
                                                       interactive = True,show_label = True,
                                                       label = "Select a category of sample questions",
                                                       elem_id = "dropdown-samples")
                        
                        
                        # iterate through the questions list
                        samples = []
                        for i,key in enumerate(QUESTIONS.keys()):
                            examples_visible = True if i == 0 else False
                            with gr.Row(visible = examples_visible) as group_examples:
                                examples_questions = gr.Examples(
                                    QUESTIONS[key],
                                    [examples_hidden],
                                    examples_per_page=8,
                                    run_on_click=False,
                                    elem_id=f"examples{i}",
                                    api_name=f"examples{i}",
                                    # label = "Click on the example question or enter your own",
                                    # cache_examples=True,
                                )
                            
                            samples.append(group_examples)
                    ##------------------- tab for Sources reporting ##------------------
                    with gr.Tab("Sources",elem_id = "tab-citations",id = 1):
                        sources_textbox = gr.HTML(show_label=False, elem_id="sources-textbox")
                        docs_textbox = gr.State("")

    def change_sample_questions(key):
        # update the questions list based on key selected
        index = list(QUESTIONS.keys()).index(key)
        visible_bools = [False] * len(samples)
        visible_bools[index] = True
        return [gr.update(visible=visible_bools[i]) for i in range(len(samples))]

    dropdown_samples.change(change_sample_questions,dropdown_samples,samples)
                        

    # static tab 'about us'
    with gr.Tab("About",elem_classes = "max-height other-tabs"):
        with gr.Row():
            with gr.Column(scale=1):
                    gr.Markdown("""The <ins>[**Office of the Auditor General (OAG)**](https://www.oag.go.ug/welcome)</ins> in Uganda, \
                consistent with the mandate of Supreme Audit Institutions (SAIs),\
                remains integral in ensuring transparency and fiscal responsibility.\
                Regularly, the OAG submits comprehensive audit reports to Parliament, \
                which serve as instrumental references for both policymakers and the public, \
                facilitating informed decisions regarding public expenditure. 
                
                However, the prevalent underutilization of these audit reports, \
                leading to numerous unimplemented recommendations, has posed significant challenges\
                to the effectiveness and impact of the OAG's operations. The audit reports made available \
                to the public have not been effectively used by them and other relevant stakeholders. \
                The current format of the audit reports is considered a challenge to the \
                stakeholders' accessibility and usability. This in one way constrains transparency \
                and accountability in the utilization of public funds and effective service delivery. 
                
                In the face of this, modern advancements in Artificial Intelligence (AI),\
                particularly Retrieval Augmented Generation (RAG) technology, \
                emerge as a promising solution. By harnessing the capabilities of such AI tools, \
                there is an opportunity not only to improve the accessibility and understanding \
                of these audit reports but also to ensure that their insights are effectively \
                translated into actionable outcomes, thereby reinforcing public transparency \
                and service delivery in Uganda. 
                
                To address these issues, the OAG has initiated several projects, \
                such as the Audit Recommendation Tracking (ART) System and the Citizens Feedback Platform (CFP). \
                These systems are designed to increase the transparency and relevance of audit activities. \
                However, despite these efforts, engagement and awareness of the audit findings remain low, \
                and the complexity of the information often hinders effective public utilization. Recognizing the need for further\
                enhancement in how audit reports are processed and understood, \
                the **Civil Society and Budget Advocacy Group (CSBAG)** in partnership with the **GIZ**, \
                has recognizing the need for further enhancement in how audit reports are processed and understood.   
                
                This prototype tool leveraging AI (Artificial Intelligence) aims at offering critical capabilities such as '
                summarizing complex texts, extracting thematic insights, and enabling interactive, \
                user-friendly analysis through a chatbot interface. By making the audit reports more accessible,\
                this aims to increase readership and utilization among stakeholders, \
                which can lead to better accountability and improve service delivery
                
                """)


    # static tab for disclaimer
    with gr.Tab("Disclaimer",elem_classes = "max-height other-tabs"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                - This chatbot is intended for specific use of answering the questions based on audit reports published by OAG, for any use beyond this scope we have no liability to response provided by chatbot.
                - We do not guarantee the accuracy, reliability, or completeness of any information provided by the chatbot and disclaim any liability or responsibility for actions taken based on its responses.
                - The chatbot may occasionally provide inaccurate or inappropriate responses, and it is important to exercise judgment and critical thinking when interpreting its output.
                - The chatbot responses should not be considered professional or authoritative advice and are generated based on patterns in the data it has been trained on.
                - The chatbot's responses do not reflect the opinions or policies of our organization or its affiliates.
                - Any personal or sensitive information shared with the chatbot is at the user's own risk, and we cannot guarantee complete privacy or confidentiality.
                - the chatbot is not deterministic, so there might be change in answer to same question when asked by different users or multiple times.
                - By using this chatbot, you agree to these terms and acknowledge that you are solely responsible for any reliance on or actions taken based on its responses.
                - **This is just a prototype and being tested and worked upon, so its not perfect and may sometimes give irrelevant answers**. If you are not satisfied with the answer, please ask a more specific question or report your feedback to help us improve the system.
                """)
            
                
                

    # using event listeners for 1. query box 2. click on example question
    # https://www.gradio.app/docs/gradio/textbox#event-listeners-arguments
    (textbox
    .submit(start_chat, [textbox, chatbot], [textbox, tabs, chatbot], queue=False, api_name="start_chat_textbox")
    # queue must be set as False (default) so the process is not waiting for another to be finished
    .then(chat, [textbox, chatbot, dropdown_sources, dropdown_reports, dropdown_category, dropdown_year], [chatbot, sources_textbox], queue=True, concurrency_limit=8, api_name="chat_textbox")
    .then(finish_chat, None, [textbox], api_name="finish_chat_textbox"))

    (examples_hidden
        .change(start_chat, [examples_hidden, chatbot], [textbox, tabs, chatbot], queue=False, api_name="start_chat_examples")
        # queue must be set as False (default) so the process is not waiting for another to be finished
        .then(chat, [examples_hidden, chatbot, dropdown_sources, dropdown_reports, dropdown_category, dropdown_year], [chatbot, sources_textbox], concurrency_limit=8, api_name="chat_examples")
        .then(finish_chat, None, [textbox], api_name="finish_chat_examples")
    )
    
    demo.queue()

demo.launch()