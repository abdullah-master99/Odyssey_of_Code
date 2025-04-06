from typing import Dict, List, Optional
import os
from crewai import Agent
from utils import (
    parse_pdf,
    chunk_text,
    generate_embedding,
    get_llm_response,
    rfp_chroma_client,
    llm,
    logger
)

class RFPAgent(Agent):
    def __init__(self):
        super().__init__(
            role="RFP Analysis Expert",
            goal="Process and analyze RFP documents using embeddings and LLM",
            backstory="I am an expert in analyzing RFP documents and answering questions about them using advanced NLP techniques.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        self._base_dir = None
        self._data_dir = None
        self._rfps_dir = None
        
        # Ensure directories exist
        os.makedirs(self.rfps_dir, exist_ok=True)

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
    def rfps_dir(self):
        if self._rfps_dir is None:
            self._rfps_dir = os.path.join(self.data_dir, 'rfps')
        return self._rfps_dir

    @property
    def collection(self):
        """Get a fresh reference to the collection"""
        return rfp_chroma_client.get_or_create_collection(
            name="rfp_documents",
            metadata={"description": "RFP document embeddings"}
        )

    def process_rfp(self, file_path: str) -> Dict:
        """Process an RFP document and store its embeddings with requirement classification"""
        logger.info(f"Processing RFP document: {file_path}")
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"RFP file not found: {file_path}")
            if not file_path.lower().endswith('.pdf'):
                raise ValueError("Only PDF files are supported")

            # Extract text
            text = parse_pdf(file_path)
            if not text:
                raise ValueError("No text could be extracted from the PDF")
            
            # Create chunks
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            logger.info(f"Created {len(chunks)} text chunks")

            # Keywords for classification
            must_have_keywords = [
                "must", "shall", "required", "mandatory", "essential",
                "necessary", "requirement", "minimum", "need to", "needs to"
            ]
            good_to_have_keywords = [
                "preferred", "optional", "desirable", "nice to have",
                "good to have", "plus", "advantage", "beneficial",
                "ideally", "preferably", "should", "may", "can"
            ]

            # Process chunks and store embeddings with classification
            for i, chunk in enumerate(chunks):
                # Classify the chunk based on keyword presence
                is_must_have = any(keyword in chunk.lower() for keyword in must_have_keywords)
                is_good_to_have = any(keyword in chunk.lower() for keyword in good_to_have_keywords)
                
                # Default to must-have if neither is detected (conservative approach)
                requirement_type = "must_have"
                if is_good_to_have and not is_must_have:
                    requirement_type = "good_to_have"

                embedding = generate_embedding(chunk)
                if embedding:
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "source": file_path,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "requirement_type": requirement_type
                        }],
                        ids=[f"rfp_{os.path.basename(file_path)}_{i}"]
                    )

            logger.info("Successfully processed and stored RFP embeddings")
            return {
                "status": "success",
                "file": file_path,
                "chunks_processed": len(chunks)
            }

        except Exception as e:
            logger.error(f"Error processing RFP: {str(e)}")
            return {
                "status": "error",
                "file": file_path,
                "error": str(e)
            }

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """Answer a question about the RFP using stored embeddings and LLM"""
        try:
            # Generate embedding for the question
            question_embedding = generate_embedding(question)
            
            # Search for relevant chunks
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=top_k
            )
            
            if not results["documents"]:
                return "I don't have enough context to answer that question about the RFP."
            
            # Combine relevant chunks
            context = "\n\n".join([doc for doc in results["documents"][0]])
            
            # Generate answer using LLM
            prompt = f"""Based on the following RFP context, please answer this question:
            
            Question: {question}
            
            RFP Context:
            {context}
            
            Provide a clear and concise answer based only on the information provided in the RFP context.
            If the information isn't available in the context, say so."""
            
            answer = get_llm_response(prompt)
            return answer

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Sorry, I encountered an error while trying to answer your question: {str(e)}"

    def execute_task(self, task, context=None, tools=None):
        """Execute RFP analysis task"""
        logger.info(f"Executing task: {task.name}")
        
        if task.name == "analyze_rfp":
            try:
                if not context or "rfp_path" not in context:
                    return {"error": "No RFP path provided"}
                
                rfp_path = context["rfp_path"]
                process_result = self.process_rfp(rfp_path)
                
                if process_result["status"] != "success":
                    return process_result
                
                # Query for different requirement types
                must_have_results = self.collection.query(
                    query_embeddings=[generate_embedding("essential mandatory required must-have needs")],
                    n_results=10,
                    where={"requirement_type": "must_have"}
                )
                
                good_to_have_results = self.collection.query(
                    query_embeddings=[generate_embedding("preferred optional good-to-have desirable advantage")],
                    n_results=10,
                    where={"requirement_type": "good_to_have"}
                )
                
                # Return structured analysis
                return {
                    "status": "success",
                    "requirements": {
                        "must_have": must_have_results["documents"][0] if must_have_results["documents"] else [],
                        "good_to_have": good_to_have_results["documents"][0] if good_to_have_results["documents"] else []
                    },
                    "total_requirements": {
                        "must_have": len(must_have_results["documents"][0]) if must_have_results["documents"] else 0,
                        "good_to_have": len(good_to_have_results["documents"][0]) if good_to_have_results["documents"] else 0
                    }
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