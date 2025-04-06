import os
from typing import Dict, List, Optional
from crewai import Agent
from utils import (
    get_llm_response, rfp_collection, company_collection, 
    generate_embedding, llm, logger,
    result_tracker
)
from .rfp_extractor_agent import RFPAgent
from .company_data_agent import CompanyDataAgent
from pydantic import Field

class EligibilityEvaluatorAgent(Agent):
    rfp_agent: RFPAgent = Field(default_factory=RFPAgent)
    company_agent: CompanyDataAgent = Field(default_factory=CompanyDataAgent)

    def __init__(self):
        super().__init__(
            role="Compliance Evaluation Expert",
            goal="Evaluate company compliance with RFP requirements",
            backstory="I am an expert in analyzing RFP compliance requirements and company capabilities to determine eligibility.",
            verbose=True,
            allow_delegation=True,
            llm=llm
        )

    def evaluate_eligibility(self, rfp_path: str, company_path: str) -> Dict:
        """
        Evaluate company compliance with RFP requirements
        """
        try:
            # Process RFP and company data
            rfp_result = self.rfp_agent.process_rfp(rfp_path)
            company_result = self.company_agent.process_company_data(company_path)

            if rfp_result["status"] == "error" or company_result["status"] == "error":
                raise ValueError("Error processing input documents")

            # Generate query embeddings for different requirement types
            core_compliance_embedding = generate_embedding("company registration US state business entity legal incorporation authorized license")
            submission_embedding = generate_embedding("submission document executive summary letter transmittal proposal attachments forms")
            additional_embedding = generate_embedding("preferred optional good-to-have nice-to-have desirable qualifications experience")

            # Get relevant sections from both documents
            core_requirements = self.rfp_agent.collection.query(
                query_embeddings=[core_compliance_embedding],
                n_results=5
            )
            
            submission_requirements = self.rfp_agent.collection.query(
                query_embeddings=[submission_embedding],
                n_results=5
            )
            
            additional_requirements = self.rfp_agent.collection.query(
                query_embeddings=[additional_embedding],
                n_results=5
            )
            
            company_info = self.company_agent.collection.query(
                query_embeddings=[core_compliance_embedding],
                n_results=5
            )

            # Prepare context for LLM evaluation
            context = {
                "core_requirements": core_requirements["documents"][0] if core_requirements["documents"] else [],
                "submission_requirements": submission_requirements["documents"][0] if submission_requirements["documents"] else [],
                "additional_requirements": additional_requirements["documents"][0] if additional_requirements["documents"] else [],
                "company_info": company_info["documents"][0] if company_info["documents"] else []
            }

            # Generate evaluation using updated prompt
            evaluation_prompt = f"""You are an expert RFP compliance evaluator. Your primary task is to determine if a company meets the basic eligibility requirements to submit a proposal.

FOCUS ON THESE POINTS FOR CORE COMPLIANCE:
1. Is the company legally registered to do business in the United States?
2. If the RFP requires registration in a specific state, is the company registered there?
3. Are there any specific jurisdictional requirements that must be met?

Analyze the provided RFP requirements and company information to determine compliance eligibility, using this EXACT format:

Core Compliance Status:
----------------------
- ELIGIBLE: [YES/NO] (Start with this, based ONLY on US/State registration requirements)
- Required US Registration: [YES/NO/NOT SPECIFIED]
  * Company Status: [COMPLIANT/NON-COMPLIANT/UNKNOWN]
- Required State Registration: [State name/NONE/NOT SPECIFIED]
  * Company Status: [COMPLIANT/NON-COMPLIANT/UNKNOWN]
- Blocking Issues: [List any blocking compliance issues]

Required Submission Documents:
---------------------------
(List ONLY documents that need to be submitted with the proposal)
- Example: Executive Summary, Forms, etc.
Note: These do not affect core compliance eligibility

Additional Desired Qualifications:
-------------------------------
(List preferred/optional qualifications that don't affect core compliance)
- Experience requirements
- Certifications that are preferred but not mandatory
- Other nice-to-have qualifications

Overall Compliance Assessment:
---------------------------
- Clear statement of ELIGIBLE or NOT ELIGIBLE
- Summary of why (focus on registration/legal requirements)
- Impact of any missing preferred qualifications

Required Actions:
---------------
1. Critical (must be completed to become eligible):
   - List actions related to core compliance issues
2. Important (needed for submission):
   - List actions related to required documents
3. Optional (for competitive advantage):
   - List actions related to preferred qualifications

RFP Core Requirements:
{context['core_requirements']}

Submission Requirements:
{context['submission_requirements']}

Additional Requirements:
{context['additional_requirements']}

Company Information:
{context['company_info']}

Remember:
1. Core eligibility depends ONLY on legal registration requirements
2. Submission documents don't affect eligibility
3. Be explicit about what makes the company eligible or not eligible
4. Separate required documents from compliance requirements"""

            evaluation_result = get_llm_response(evaluation_prompt)

            return {
                "status": "success",
                "evaluation": evaluation_result,
                "rfp_analysis": rfp_result,
                "company_analysis": company_result,
                "is_compliant": self._check_compliance(evaluation_result)
            }

        except Exception as e:
            logger.error(f"Error in compliance evaluation: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def execute_task(self, task, context=None, tools=None):
        """Execute compliance evaluation task"""
        logger.info(f"Executing task: {task.name}")
        
        if task.name == "evaluate_eligibility":
            try:
                if not context or "rfp_path" not in context or "company_path" not in context:
                    return {"error": "RFP and company paths are required"}
                
                evaluation_result = self.evaluate_eligibility(
                    context["rfp_path"], 
                    context["company_path"]
                )
                
                if evaluation_result["status"] == "error":
                    return evaluation_result
                
                # Structure the result
                return {
                    "core_compliance": self._format_core_compliance(evaluation_result),
                    "submission_requirements": self._format_submission_requirements(evaluation_result),
                    "additional_qualifications": self._format_additional_qualifications(evaluation_result),
                    "compliance_assessment": self._format_compliance_assessment(evaluation_result),
                    "required_actions": self._format_required_actions(evaluation_result),
                    "is_compliant": evaluation_result.get("is_compliant", False)
                }
                
            except Exception as e:
                logger.error(f"Error in evaluate_eligibility task: {str(e)}")
                return {"error": str(e)}
        
        return {"error": f"Unknown task: {task.name}"}

    def _format_core_compliance(self, result):
        """Format core compliance status"""
        try:
            return result["evaluation"].split("Core Compliance Status:")[1].split("Required Submission Documents:")[0].strip()
        except:
            return "No core compliance analysis available"

    def _format_submission_requirements(self, result):
        """Format submission requirements"""
        try:
            return result["evaluation"].split("Required Submission Documents:")[1].split("Additional Desired Qualifications:")[0].strip()
        except:
            return "No submission requirements available"

    def _format_additional_qualifications(self, result):
        """Format additional qualifications analysis"""
        try:
            return result["evaluation"].split("Additional Desired Qualifications:")[1].split("Overall Compliance Assessment:")[0].strip()
        except:
            return "No additional qualifications analysis available"

    def _format_compliance_assessment(self, result):
        """Format overall compliance assessment"""
        try:
            return result["evaluation"].split("Overall Compliance Assessment:")[1].split("Required Actions:")[0].strip()
        except:
            return "No compliance assessment available"

    def _format_required_actions(self, result):
        """Format required actions"""
        try:
            actions_section = result["evaluation"].split("Required Actions:")[1]
            actions = [action.strip() for action in actions_section.split('\n') if action.strip()]
            return actions
        except:
            return []

    def _check_compliance(self, evaluation_result: str) -> bool:
        """Check if the company is compliant based on core compliance requirements"""
        try:
            core_status = evaluation_result.split("Core Compliance Status:")[1].split("Required Submission Documents:")[0].lower()
            return "eligible: yes" in core_status and "eligible: no" not in core_status
        except:
            return False