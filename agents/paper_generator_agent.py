import os
from typing import Dict
from crewai import Agent
from utils import (
    get_llm_response,
    generate_embedding,
    llm,
    logger
)
from .subject_pdf_agent import SubjectPDFAgent
from .question_paper_agent import QuestionPaperAgent
from pydantic import Field

class TeacherAssistantAgent(Agent):
    subject_agent: SubjectPDFAgent = Field(default_factory=SubjectPDFAgent)
    question_paper_agent: QuestionPaperAgent = Field(default_factory=QuestionPaperAgent)

    def __init__(self):
        super().__init__(
            role="Teacher Assistant",
            goal="Generate a new question paper based on subject content and an existing question paper template.",
            backstory="I assist teachers in creating question papers by analyzing subject content and reusing marking schemes from existing templates.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    def generate_question_paper(self, subject_pdf_path: str, question_paper_template_path: str) -> Dict:
        try:
            # Process the subject content and the question paper template
            subject_result = self.subject_agent.process_subject_pdf(subject_pdf_path)
            template_result = self.question_paper_agent.process_question_paper(question_paper_template_path)

            if subject_result["status"] == "error":
                raise ValueError(f"Error processing subject PDF: {subject_result.get('error')}")
            if template_result["status"] == "error":
                raise ValueError(f"Error processing question paper template: {template_result.get('error')}")

            # Extract relevant information from the subject content and template
            subject_content = subject_result.get("content", "No content found.")
            template_questions = template_result.get("questions", [])
            marking_scheme = template_result.get("marking_scheme", "No marking scheme found.")

            # Use LLM to generate a new question paper based on the template and subject content
            prompt = f"""
            You are an expert exam setter. Based on the following subject content and the structure of the provided question paper template, generate a new question paper:
            
            Subject Content:
            {subject_content}
            
            Template Questions:
            {template_questions}
            
            Marking Scheme:
            {marking_scheme}
            
            The new question paper should include:
            - Multiple-choice questions
            - Short-answer questions
            - Long-answer questions
            - Case studies
            - Diagrams (if applicable)
            Ensure the questions cover all key topics from the subject content and follow the structure and marking scheme of the template.
            """
            generated_questions = get_llm_response(prompt)

            # Parse the LLM response into structured sections
            generated_paper = {
                "multiple_choice": generated_questions.get("multiple_choice", []),
                "short_answer": generated_questions.get("short_answer", []),
                "long_answer": generated_questions.get("long_answer", []),
                "case_studies": generated_questions.get("case_studies", []),
                "diagrams": generated_questions.get("diagrams", []),
                "total_questions": len(generated_questions.get("multiple_choice", [])) +
                                len(generated_questions.get("short_answer", [])) +
                                len(generated_questions.get("long_answer", [])) +
                                len(generated_questions.get("case_studies", [])) +
                                len(generated_questions.get("diagrams", [])),
                "difficulty_distribution": {"easy": 2, "medium": 3, "hard": 1},  # Example distribution
                "topics_covered": subject_result.get("topics", [])
            }

            return {
                "status": "success",
                "question_paper": generated_paper,
                "subject_analysis": subject_result,
                "template_analysis": template_result
            }

        except Exception as e:
            logger.error(f"Error generating question paper: {str(e)}")
            return {"status": "error", "message": str(e)}

    def execute_task(self, task, context=None, tools=None):
        """Execute question paper generation task"""
        logger.info(f"Executing task: {task.name}")
        
        if task.name == "generate_question_paper":
            try:
                if not context or "subject_pdf_path" not in context or "question_paper_template_path" not in context:
                    return {"error": "Both subject PDF path and question paper template path are required"}
                
                generation_result = self.generate_question_paper(
                    context["subject_pdf_path"],
                    context["question_paper_template_path"]
                )
                return generation_result

            except Exception as e:
                logger.error(f"Error executing generate_question_paper task: {str(e)}")
                return {"error": str(e)}
        
        return {"error": f"Unknown task: {task.name}"}