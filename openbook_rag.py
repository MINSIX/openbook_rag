

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)



from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 기본 모델을 로드합니다

from peft import PeftModel
# 학습모델의 PEFT



model = AutoModelForCausalLM.from_pretrained("LDCC/LDCC-SOLAR-10.7B", load_in_8bit=False,
    torch_dtype=torch.float16, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("LDCC/LDCC-SOLAR-10.7B")

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.chains import LLMChain

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.9,
    max_new_tokens=1024,
    return_full_text=False,
    
)

prompt_template = """
### [INST]
Instruction: Answer the question based on your knowledge. 
너는 지금부터 한국사 오픈북 시험을 볼꺼야. 내가하는 질문에 대해 정보를 기반으로 구체적이고 상세하게 답변을 해줘. 
쓸대없는 말은 넣지 말아줘. 매우 매우 구체적으로 설명해줘.


Here is context to help:

{context}

### QUESTION:
{question}

[/INST]
 """

koplatyi_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=koplatyi_llm, prompt=prompt)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.schema.runnable import RunnablePassthrough
# loader = PyPDFLoader("./data/4. Network Layer Data Plane 1.pdf")
import pandas as pd
start_time = time.time()
 


# # Convert each column to a list
from langchain.document_loaders import PyPDFLoader
import os



texts=[]
# PDF 파일을 로드하여 페이지 단위로 텍스트를 추출하는 함수
def load_pdf_texts(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(pages)

# "./data" 디렉토리에서 모든 PDF 파일을 가져와서 texts 리스트에 추가
data_folder = "./data"

for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(data_folder, filename)
        texts.extend(load_pdf_texts(pdf_path))



from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(texts, hf)
end_time = time.time()
execution_time = end_time - start_time
print(f"응답 시간: {execution_time} s")
retriever = db.as_retriever(
                            search_type="similarity",
                            search_kwargs={'k':5,
                             },
                        )
rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)
import warnings
warnings.filterwarnings('ignore')

while True:
   
    user_input = input("검색어를 입력하세요 (종료하려면 'xxx'를 입력하세요): ")

    if user_input.lower() == 'xxx':
        print("프로그램을 종료합니다.")
        break
    start_time = time.time()    
    result = rag_chain.invoke(user_input)
    
    for i in result['context']:
        print(f"주어진 근거: {i.page_content} / 출처: {i.metadata['source']} - {i.metadata['page']} \n\n")

    print(f"\n답변: {result['text']}")