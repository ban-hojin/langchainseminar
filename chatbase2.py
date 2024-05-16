from openai import OpenAI
import streamlit as st

st.title("My Friend")

client = OpenAI(api_key="sk-")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

system_message = '''
너는 나의 오래된 친구야. 나에게 항상 속 싶은 대화하는 일은 하지.
항상 나에게 오래된 친구처럼 대하고 서로 진실되게 말하도록 하자.
영어로 질문을 받아도 무조건 한글로 답변해줘.
한글이 아닌 답변일 때는 다시 생각해서 꼭 한글로 만들어줘
모든 답변 끝에 서로 격려하는 이모티콘을 추가해줘
'''

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.session_state.messages = [{"role": "system", "content": system_message}]

for idx, message in enumerate(st.session_state.messages):
    if idx > 0:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})