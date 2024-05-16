import os #운영체제(os) 모듈을 가져올 때 사용되는 라이브러리
os.environ["OPENAI_API_KEY"] = "sk-" #openai 키 입력

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# 네이버 뉴스기사 주소
url = 'https://www.sedaily.com/NewsView/2D93HB80SL'

# 웹 문서 크롤링
loader = WebBaseLoader(url)

# 뉴스기사의 본문을 Chunk 단위로 쪼갬
text_splitter = CharacterTextSplitter(        
    separator="\n\n",
    chunk_size=3000,     # 쪼개는 글자수
    chunk_overlap=300,   # 오버랩 글자수
    length_function=len,
    is_separator_regex=False,
)

# 웹사이트 내용 크롤링 후 Chunk 단위로 분할
docs = WebBaseLoader(url).load_and_split(text_splitter)

# 각 Chunk 단위의 템플릿
template = '''다음의 내용을 한글로 요약해줘:

{text}
'''

# 전체 문서(혹은 전체 Chunk)에 대한 지시(instruct) 정의
combine_template = '''{text}

요약의 결과는 다음의 항목으로 작성해줘:
제목: 기사의 제목
주요내용: 한 줄로 요약된 내용
작성자: 홍길동 연구원
내용: 주요내용을 블랫포인트 형식으로 작성
'''

# 템플릿 생성
prompt = PromptTemplate(template=template, input_variables=['text'])
combine_prompt = PromptTemplate(template=combine_template, input_variables=['text'])

# LLM 객체 생성
llm = ChatOpenAI(temperature=0, 
                 model_name='gpt-3.5-turbo-16k')

# 요약을 도와주는 load_summarize_chain
chain = load_summarize_chain(llm, 
                             map_prompt=prompt, 
                             combine_prompt=combine_prompt, 
                             chain_type="map_reduce", 
                             verbose=False)



print(chain.run(docs))