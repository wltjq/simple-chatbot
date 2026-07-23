import streamlit as st
from llm import get_ai_message
from dotenv import load_dotenv
load_dotenv() 

st.set_page_config(page_title="Chatbot")

st.title("운수좋은 날 Chatbot")
st.caption("운수좋은 날에 관련된 모든 것을 답해드립니다.")

#대화 기록을 세션 상태에 저장(재시작하면 초기화)
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

#지금까지의 chat history를 화면에 순서대로 출력
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="궁금한 내용들을 말씀해주세요!"):
    #사용자 메시지를 화면에 표시하고 기록
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role":"user","content":user_question})

    #AI 응답 생성 중 로딩 스피너 표시
    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_message(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role":"ai","content":ai_message})



