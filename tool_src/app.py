import streamlit as st
import requests
import json
import nltk
import time
import os
os.chdir('../')
import src.end2end.summarize_tool as summary_tool
def callback():
	st.session_state.submit_button = True
	st.session_state.title = None

def call_API_Question(query):
    
    # count the page
    i = 1
    # the list of answer
    answer_list = []
    paras = {
        'pagesize': 99,
        'order': 'desc',
        'sort': 'relevance',
        'q': str(query),
        'site': 'stackoverflow',
        # page start from 1
        'page': '1', 
        'answers': '1',
        'key':'tFaahBz1)Kq70INbmCkYrw((',
    }


    r = requests.get('https://api.stackexchange.com/2.3/search/advanced', params=paras).json()

    answer_list=r['items']

    # if len(answer_list)<99:
        # st.text("The number of relevant questions: " + str(len(answer_list)))
    # else:
        # st.text("The number of relevant questions: 100+")
    return answer_list

def call_API_Answer(question_id, num_answers):

    
    # count the page
    i = 1
    # the list of answer
    answer_list = []
    paras = {
        'order': 'desc',
        'sort': 'votes',
        'site': 'stackoverflow',
        'pagesize': num_answers,
        'key':'tFaahBz1)Kq70INbmCkYrw((',
        "filter":"!amZQw(pwL2DHTe",
    }

    # print("query: ", paras['q'])

    r = requests.get('https://api.stackexchange.com/2.3/questions/'+question_id[:-1]+'/answers', params=paras).json()
    # print('https://api.stackexchange.com/2.3/questions/'+question_id+'/answers')

    answer_list=r['items']

    # print('num of quotation is: ', r['quota_remaining'])


    # print('the number of answers: ', len(answer_list))

    return answer_list

    

def split_data(test_unit):

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(test_unit, 'html.parser')

    result = []
    output= []
    for s in soup('table'):
        s.string = '[table]'
        s.name = 'p'

    for s in soup('strong'):
        s.unwrap()

    for div in soup.find_all("div", {'class':'snippet'}): 
        div.string='[code snippet]'
        div.name = 'p'


    for s in soup('pre'):
        if s.code:
            s.code.unwrap()
        s.string = '[code snippet]'
        s.name = 'p'


    for s in soup('a'):
        hyper_link = s.get('href')
        # s.string='['+s.get_text()+']('+hyper_link+')'
        s.string='['+s.get_text()+' (hyper-link)]'
        try:
            s.unwrap()
        except:
            print('error')

    for s in soup('img'):
        s.string = '[image]'
        s.unwrap()

    for s in soup('li'):
        s.string = s.get_text()
        s.name = 'p'


    # for p in soup('p'):
    #     result.append(p.get_text()+"\n Paragraph end")

    # for item in result:
    #     output+=nltk.sent_tokenize(item)

    for p in soup('p'):
        output+=nltk.sent_tokenize(p.get_text())
        # output.append("Paragraph end")
    return output


def answer_selection(branch_url, num_questions, num_answers):
    question_list = call_API_Question(branch_url)
    question_ids = [question_list[i]['question_id'] for i in range(len(question_list))]
    # st.text('Extracting the most relevant answers...')
    print('num of questions: ', len(question_ids))
    answer_str = ''

    # get the top-'num_questions' questions
    if len(question_ids) <num_questions:
        for question_id in question_ids:
            answer_str+= str(question_id)+';'
    if len(question_ids) >= num_questions:
        for question_id in question_ids[:num_questions]:
            answer_str+= str(question_id)+';'
    # print('answer_str: ', answer_str)
    # st.text('Extracting the most relevant %d questions...'%(int(num_questions)))
    # st.text('Extracting the most voted %d answers from top-%d questions...'%(int(num_answers),int(num_questions)))

    answer_list = call_API_Answer(answer_str, num_answers)

    print('num of answers: ', len(answer_list))

    sent_dic = {}

    sent_list = []
    answers = {}
    for answer in answer_list:
        # print(answer)
        answer_id = answer['answer_id']
        answer_body = split_data(answer['body'])
        sent_dic[answer_id] = answer_body
        sent_list.extend(answer_body)
        answers[answer_id]=answer_body
        
    return sent_list, answers


    # answer_ids = ''
    # for answer in answer_list:
    #     answer_ids += str(answer['answer_id'])+';'
    
    # extracting answer content, input {ids}
    # answer_content = call_Answer_Body(answer_ids)

    # print(answer_content[0])

def config():
    # st.set_page_config(layout="wide")

    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)

    hide_default_format = """
       <style>
       MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    

def print_url(url):
    st.markdown(url, unsafe_allow_html=True)

def main():
    config()
    st.title("TECHSUMBOT")
    st.subheader("A Stack Overflow Answer Summarization Tool")
    menu=['Home', 'Guide','Advance Summarization']
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=='Home':
        # st.subheader("Home")
        st.markdown('<p class="big-font">This is a answer summarization app for Stack Overflow. Please select the Guide menu to learn the basic usage of TECHSUMBOT.\
            For customized answer summarization (e.g., control of summary length, number of relevant answers), please select the advance summarization option.\
                </p>', unsafe_allow_html=True)
        # st.markdown('.')
        # st.text("Please select the Guide menu to see how to use this app for future details")
        st.markdown("Please input your query in the search bar below:")
        with st.form(key='form1'):
            branch_url = st.text_input("Enter the query", placeholder="the query you want to search")
            submit_button = st.form_submit_button(label='Generate the answer summary', on_click=callback)
        if "submit_button" not in st.session_state:
            st.session_state.submit_button = False
        else:
            submit_button = True
        if submit_button:
            sent_list,answer_list = answer_selection(branch_url, num_questions=10, num_answers=10)
            print('sent_list: ', sent_list)
            st.subheader('The answer summarization is shown below:')
            summary = summary_tool(sent_list, branch_url)[:5]
            # for sent in summary:
                # st.markdown(sent)

            for num, sent in enumerate(summary):
                st.markdown(sent)
                # c1.write(sent)

                for answer in answer_list.keys():
                    if sent in answer_list[answer]:
                        url = "https://stackoverflow.com/a/" + str(answer)
                        # clicked = st.button("SO answer link")
                        
                        if st.button('original answer '+str(num), on_click=print_url, args = [url]):                            
                            st.write(url)

                    continue   




    elif choice=='Guide':
        st.subheader("Usage")
        st.markdown("To use TECHSUMBOT, you should input your technical question in the search bar. \
            Please note that currently we only support the technical questions in software engineering domain.\n  \
            For example, you can input the following questions:")
        st.text('1. How to use **One specific API** in Python?')
        st.markdown("Our tool would extract the most relevant answers from Stack Overflow and return an answer summary to you.\n \
            The default setting is to extract the top-10 most relevant questions and the top-10 most voted answers from each question.\
            The answer summarization consist of five answer sentences by default.\n\
            You can also click the 'Advance Summarization' menu to control the number of relevant answers for summarization and length of the summaries.")

        st.subheader("Advanced Summarization")
        st.markdown('To control the number of relevant answers for summarization and length of the summaries, please click the "Advance Summarization" menu.')
        st.markdown('You are allowed to input the number of relevant answers for summarization and length of the summaries in each form, and then input the query in the search bar.')
        st.markdown('The default setting is to extract the top-10 most relevant questions and the top-10 most voted answers from each question. The default answer summarization consist of five answer sentences.')
        st.markdown('TECHSUMBOT would generate an answer summary that meets your requirements.')

        st.subheader("Summarization Algorithm")
        st.markdown("Please refes to our published paper for the details of the summarization algorithm.")
        st.text("Answer Summarization for Technical Queries: Benchmark and New Approach ASE 2022\
            https://arxiv.org/abs/2209.10868")


    elif choice=='Advance Summarization':
        
        # bottom 2
        st.markdown("Please input your expected summary length:")
        with st.form(key='form2'):
            length_sum = st.text_input("Enter the number of sentences", placeholder="the length of the summary")
            submit_button_2 = st.form_submit_button(label='Confirm', on_click=callback)
        if "submit_button_2" not in st.session_state:
            st.session_state.submit_button = False
        else:
            submit_button_2 = True

        # bottom 3
        st.markdown("Please input your summarization scope:")
        st.markdown("The summarization scope refers to the number of the most relevant answers to your technical query. By default we set the number as 10.")
        with st.form(key='form3'):
            length_answer = st.text_input("Enter the number of answers", placeholder="the number of the answers")
            submit_button_3 = st.form_submit_button(label='Confirm', on_click=callback)
            # length_question = st.text_input("Enter the number of answers", placeholder="the number of the answers")
            # submit_button_4 = st.form_submit_button(label='Generate the answer summary', on_click=callback)            
        if "submit_button_3" not in st.session_state:
            st.session_state.submit_button = False
        else:
            submit_button_3 = True

        # bottom 1
        st.markdown("Please input your query in the search bar below:")
        with st.form(key='form1'):
            branch_url = st.text_input("Enter the query", placeholder="the query you want to search")
            submit_button_1 = st.form_submit_button(label='Generate the answer summary', on_click=callback)
        if "submit_button_1" not in st.session_state:
            st.session_state.submit_button = False
        else:
            submit_button_1 = True

        option = st.selectbox(
    'Programming Language',
    ('Python', 'Java', 'General'))

        if submit_button_1:
            print('test')
            print(length_sum)
            print(length_answer)
            if not length_answer:
                length_answer = 10
            if not length_sum:
                length_sum = 5
            sent_list,answer_list = answer_selection(branch_url, num_questions=10, num_answers=int(length_answer))
            st.subheader('The answer summarization is shown below:')
            summary = summary_tool(sent_list, branch_url)[:int(length_sum)]


            # c1, c2  = st.columns((5, 1))


            for num, sent in enumerate(summary):
                st.markdown(sent)
                # c1.write(sent)

                for answer in answer_list.keys():
                    if sent in answer_list[answer]:
                        url = "https://stackoverflow.com/a/" + str(answer)
                        # clicked = st.button("SO answer link")
                        
                        if st.button('original answer '+str(num), on_click=print_url, args = [url]):                            
                            st.write(url)

                    continue


    elif choice=='About':
        st.subheader("About")
        st.text("This tool is an implementation for paper TechSumBot")

if __name__ =="__main__":
    main()
    