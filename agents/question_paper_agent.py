from typing import Dict, List, Optional
import os
from crewai import Agent
from utils import (
    parse_pdf,
    chunk_text,
    generate_embedding,
    get_llm_response,
    company_chroma_client,
    llm,
    logger
)

class QuestionPaperAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Question Paper Analysis Expert",
            goal="Process and analyze question papers using embeddings and LLM",
            backstory="I am an expert in analyzing question paper content and information using advanced NLP techniques.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        self._base_dir = None
        self._data_dir = None
        self._question_paper_dir = None
        
        # Ensure directories exist
        os.makedirs(self.question_paper_dir, exist_ok=True)

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
    def question_paper_dir(self):
        if self._question_paper_dir is None:
            self._question_paper_dir = os.path.join(self.data_dir, 'question_papers')
        return self._question_paper_dir

    @property
    def collection(self):
        """Get a fresh reference to the collection"""
        return company_chroma_client.get_or_create_collection(
            name="question_papers",
            metadata={"description": "Question paper embeddings"}
        )

    def process_question_paper(self, file_path: str) -> Dict:
        """Process a question paper and store its embeddings"""
        logger.info(f"Processing question paper: {file_path}")
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Question paper file not found: {file_path}")
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
                        ids=[f"question_paper_{os.path.basename(file_path)}_{i}"]
                    )

            logger.info("Successfully processed and stored question paper embeddings")
            return {
                "status": "success",
                "file": file_path,
                "chunks_processed": len(chunks)
            }

        except Exception as e:
            logger.error(f"Error processing question paper: {str(e)}")
            return {
                "status": "error",
                "file": file_path,
                "error": str(e)
            }

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """Answer a question based on the uploaded question papers"""
        try:
            # Generate embedding for the question
            question_embedding = generate_embedding(question)
            
            # Search for relevant chunks
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=top_k
            )
            
            if not results["documents"]:
                return "I don't have enough context to answer that question based on the uploaded question papers."
            
            # Combine relevant chunks
            context = "\n\n".join([doc for doc in results["documents"][0]])
            
            # Generate answer using LLM
            prompt = f"""Based on the following question paper data, please answer this question:
            
            Question: {question}
            
            Question Paper Context:
            {context}
            
            Provide a clear and concise answer based only on the information provided in the question paper context.
            If the information isn't available in the context, say so."""
            
            answer = get_llm_response(prompt)
            return answer

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Sorry, I encountered an error while trying to answer your question: {str(e)}"

    def get_question_paper_stats(self) -> Dict:
        """Get statistics about processed question papers"""
        try:
            return {
                "total_documents": len(self.collection.get()["ids"]),
                "document_sources": list(set([
                    meta["source"] for meta in self.collection.get()["metadatas"]
                ]))
            }
        except Exception as e:
            logger.error(f"Error getting question paper stats: {str(e)}")
            return {"error": str(e)}

    def execute_task(self, task, context=None, tools=None):
        """Execute question paper analysis task"""
        logger.info(f"Executing task: {task.name}")
        
        if task.name == "analyze_question_paper":
            try:
                if not context or "question_paper_path" not in context:
                    return {"error": "No question paper path provided"}
                
                question_paper_path = context["question_paper_path"]
                process_result = self.process_question_paper(question_paper_path)
                
                if process_result["status"] != "success":
                    return process_result
                
                # Return structured analysis
                return {
                    "status": "success",
                    "total_sections": len(process_result["chunks_processed"])
                }
                
            except Exception as e:
                logger.error(f"Task execution failed: {str(e)}")
                return {"error": f"Task execution failed: {str(e)}"}
                
        elif task.name == "answer_question":
            if not context or "question" not in context:
                return {"error": "No question provided"}
            
            answer = self.answer_question(context["question"])
            return {"answer": answer}
            
        elif task.name == "get_question_paper_stats":
            return self.get_question_paper_stats()
            
        else:
            return {"error": f"Unknown task: {task.name}"}