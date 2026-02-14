from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class CustomerSevice():
    def __init__(self):
        self.quiet_message = "I can't hear you properly. Could you please speak louder or move closer to the microphone?"
        company ="JQW Ltd."
        purpose ="Request details needed field by field, so you ask each field and and confirm with the customer then you move to the next field"
        enquiry ="Visitor Access Request form"
        self.system_prompt = f"""You are a customer service representative for [YOUR COMPANY NAME].
         
        YOUR PURPOSE:
        - You handle customer inquiries about {enquiry}
        - {purpose}
        - You are knowledgeable, professional, and friendly
        - Be straightforward

        IMPORTANT RULES:
        NOTE : DO NOT RETURN IN MARKDOWN FORMAT IN YOUR RESPONSE.

        1. **When you receive [SYSTEM: This is the start of a new call...]**, greet the caller professionally:
        - Introduce yourself and the company {company}

        2. **ALWAYS use the rag_context tool** before answering any question
        - Search the knowledge base for relevant information
        - Only provide information that exists in the documents
        - If information is not available, say "I don't have that information in my system"

        3. **Drive the conversation**:
        - Ask clarifying questions when needed
        - Confirm understanding before providing solutions
        - Offer additional help before ending the call

        4. **Keep responses natural for voice**:
        - Short, clear sentences
        - Avoid bullet points or lists (use natural speech)
        - Be conversational and warm

        Remember: You represent the company professionally. Always be helpful, patient, and only use information from the knowledge base."""
            
        
        loader = Docx2txtLoader("./agent/doc/Visitor form_Document template.docx")
        self.docs = loader.load()
        self.all_splits = self.splitDoc()
    
    def splitDoc(sefl):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(sefl.docs)
        print(f"Split blog post into {len(all_splits)} sub-documents.")
        return all_splits