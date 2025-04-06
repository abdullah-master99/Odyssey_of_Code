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

class CompanyDataAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Company Data Analysis Expert",
            goal="Process and analyze company data documents using embeddings and LLM",
            backstory="I am an expert in analyzing company capabilities and experience using advanced NLP techniques.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        self._base_dir = None
        self._data_dir = None
        self._company_data_dir = None
        
        # Ensure directories exist
        os.makedirs(self.company_data_dir, exist_ok=True)

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
    def company_data_dir(self):
        if self._company_data_dir is None:
            self._company_data_dir = os.path.join(self.data_dir, 'company_data')
        return self._company_data_dir

    @property
    def collection(self):
        """Get a fresh reference to the collection"""
        return company_chroma_client.get_or_create_collection(
            name="company_documents",
            metadata={"description": "Company document embeddings"}
        )

    def process_company_data(self, file_path: str) -> Dict:
        """Process a company data document and store its embeddings"""
        logger.info(f"Processing company document: {file_path}")
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Company data file not found: {file_path}")
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
                        ids=[f"company_{os.path.basename(file_path)}_{i}"]
                    )

            logger.info("Successfully processed and stored company data embeddings")
            return {
                "status": "success",
                "file": file_path,
                "chunks_processed": len(chunks)
            }

        except Exception as e:
            logger.error(f"Error processing company data: {str(e)}")
            return {
                "status": "error",
                "file": file_path,
                "error": str(e)
            }

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """Answer a question about the company using stored embeddings and LLM"""
        try:
            # Generate embedding for the question
            question_embedding = generate_embedding(question)
            
            # Search for relevant chunks
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=top_k
            )
            
            if not results["documents"]:
                return "I don't have enough context to answer that question about the company."
            
            # Combine relevant chunks
            context = "\n\n".join([doc for doc in results["documents"][0]])
            
            # Generate answer using LLM
            prompt = f"""Based on the following company data, please answer this question:
            
            Question: {question}
            
            Company Context:
            {context}
            
            Provide a clear and concise answer based only on the information provided in the company context.
            If the information isn't available in the context, say so."""
            
            answer = get_llm_response(prompt)
            return answer

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Sorry, I encountered an error while trying to answer your question: {str(e)}"

    def get_company_stats(self) -> Dict:
        """Get statistics about processed company data"""
        try:
            return {
                "total_documents": len(self.collection.get()["ids"]),
                "document_sources": list(set([
                    meta["source"] for meta in self.collection.get()["metadatas"]
                ]))
            }
        except Exception as e:
            logger.error(f"Error getting company stats: {str(e)}")
            return {"error": str(e)}

    def execute_task(self, task, context=None, tools=None):
        """Execute company data analysis task"""
        logger.info(f"Executing task: {task.name}")
        
        if task.name == "analyze_company":
            try:
                if not context or "company_path" not in context:
                    return {"error": "No company data path provided"}
                
                company_path = context["company_path"]
                process_result = self.process_company_data(company_path)
                
                if process_result["status"] != "success":
                    return process_result
                
                # Define capability categories to analyze
                categories = {
                    "technical": generate_embedding("technical skills expertise competencies technologies tools"),
                    "experience": generate_embedding("experience past projects track record history achievements"),
                    "certifications": generate_embedding("certifications licenses accreditations compliance standards"),
                    "team": generate_embedding("team personnel staff resources capacity expertise"),
                    "infrastructure": generate_embedding("infrastructure facilities equipment capabilities systems")
                }
                
                capabilities = {}
                for category, query_embedding in categories.items():
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5
                    )
                    capabilities[category] = results["documents"][0] if results["documents"] else []
                
                # Return structured analysis
                return {
                    "status": "success",
                    "capabilities": capabilities,
                    "total_sections": len(process_result["chunks_processed"]),
                    "categories_analyzed": list(capabilities.keys())
                }
                
            except Exception as e:
                logger.error(f"Task execution failed: {str(e)}")
                return {"error": f"Task execution failed: {str(e)}"}
                
        elif task.name == "answer_question":
            if not context or "question" not in context:
                return {"error": "No question provided"}
            
            answer = self.answer_question(context["question"])
            return {"answer": answer}
            
        elif task.name == "get_company_stats":
            return self.get_company_stats()
            
        else:
            return {"error": f"Unknown task: {task.name}"}