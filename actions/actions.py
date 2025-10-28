from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from ollama import ChatModel  # type: ignore

class ActionRAGAnswer(Action):
    def name(self):
        return "action_rag_answer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):

        # 1️⃣ Get the latest user message
        user_query = tracker.latest_message.get('text')

        # 2️⃣ Initialize the Ollama LLM
        llm = ChatModel(model="mistral")

        # 3️⃣ Initialize Chroma vectorstore with your docs
        # This assumes you already persisted your embeddings in "db"
        retriever = Chroma(persist_directory="db")

        # 4️⃣ Build the RAG QA chain
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever.as_retriever())

        # 5️⃣ Run query
        answer = llm.predict(user_query)

        # 6️⃣ Send the answer back to Rasa
        dispatcher.utter_message(text=answer)
        return []
