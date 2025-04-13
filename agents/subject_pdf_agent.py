from typing import Dict, List, Optional
import os
from crewai import Agent
from utils import (
    parse_pdf,
    chunk_text,
    generate_embedding,
    get_llm_response,
    rfp_chroma_client,  # If you have a dedicated client for chapters/subject PDFs, update this import accordingly.
    llm,
    logger
)

class SubjectPDFAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Subject PDF Analysis Expert",
            goal="Process and analyze chapter or subject PDF documents using embeddings and LLM",
            backstory="I specialize in analyzing chapters or subject PDFs and answering questions based on their content using advanced NLP techniques.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        self._base_dir = None
        self._data_dir = None
        self._subject_pdfs_dir = None
        
        # Ensure directories exist
        os.makedirs(self.subject_pdfs_dir, exist_ok=True)

    @property
    def base_dir(self):
        if self._base_dir is None:
            self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return self._base_dir

    @property
    def data_dir(self):
        if self._data_dir is None:
            self._data_dir = os.path.join(self.base_dir, 'data')
        return self._data_dir

    @property
    def subject_pdfs_dir(self):
        if self._subject_pdfs_dir is None:
            self._subject_pdfs_dir = os.path.join(self.data_dir, 'subject_pdfs')
        return self._subject_pdfs_dir

    @property
    def collection(self):
        """Get a fresh reference to the collection"""
        return rfp_chroma_client.get_or_create_collection(
            name="subject_documents",
            metadata={"description": "Subject/chapter document embeddings"}
        )

    def process_subject_pdf(self, file_path: str) -> Dict:
        """Process a chapter or subject PDF and store its embeddings"""
        logger.info(f"Processing subject PDF document: {file_path}")
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Subject PDF file not found: {file_path}")
            if not file_path.lower().endswith('.pdf'):
                raise ValueError("Only PDF files are supported")

            # Extract text
            text = parse_pdf(file_path)
            if not text:
                raise ValueError("No text could be extracted from the PDF")
            
            # Create chunks
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            logger.info(f"Created {len(chunks)} text chunks")

            # Process chunks and store embeddings
            for i, chunk in enumerate(chunks):
                embedding = generate_embedding(chunk)
                if embedding:
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "source": file_path,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }],
                        ids=[f"subject_{os.path.basename(file_path)}_{i}"]
                    )

            logger.info("Successfully processed and stored subject PDF embeddings")
            return {
                "status": "success",
                "file": file_path,
                "chunks_processed": len(chunks)
            }

        except Exception as e:
            logger.error(f"Error processing subject PDF: {str(e)}")
            return {
                "status": "error",
                "file": file_path,
                "error": str(e)
            }

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """Answer a question about the subject/chapter using stored embeddings and LLM"""
        try:
            # Generate embedding for the question
            question_embedding = generate_embedding(question)
            
            # Search for relevant chunks
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=top_k
            )
            
            if not results["documents"]:
                return "I don't have enough context to answer that question based on the provided subject PDF."
            
            # Combine relevant chunks
            context = "\n\n".join([doc for doc in results["documents"][0]])
            
            # Generate answer using LLM
            prompt = f"""Based on the following subject PDF content, please answer this question:

Question: {question}

Subject PDF Context:
{context}

Provide a clear and concise answer based only on the information provided in the subject PDF context.
If the information isn't available in the context, say so."""
            
            answer = get_llm_response(prompt)
            return answer

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Sorry, I encountered an error while trying to answer your question: {str(e)}"

    def execute_task(self, task, context=None, tools=None):
        """Execute subject PDF analysis task"""
        logger.info(f"Executing task: {task.name}")
        
        if task.name == "analyze_subject_pdf":
            try:
                if not context or "subject_pdf_path" not in context:
                    return {"error": "No subject PDF path provided"}
                
                subject_pdf_path = context["subject_pdf_path"]
                process_result = self.process_subject_pdf(subject_pdf_path)
                
                if process_result["status"] != "success":
                    return process_result
                
                # Return structured analysis
                return {
                    "status": "success",
                    "chunks_processed": process_result["chunks_processed"]
                }
                
            except Exception as e:
                logger.error(f"Task execution failed: {str(e)}")
                return {"error": f"Task execution failed: {str(e)}"}
                
        elif task.name == "answer_question":
            if not context or "question" not in context:
                return {"error": "No question provided"}
            
            answer = self.answer_question(context["question"])
            return {"answer": answer}
            
        else:
            return {"error": f"Unknown task: {task.name}"}