import os
import logging
from utils import logger, reset_collections
from agents.master_agent import EligibilityEvaluatorAgent

def test_evaluation():
    """Test the eligibility evaluation workflow"""
    # Test files
    rfp_path = os.path.join("data", "rfps", "ELIGIBLE_RFP_-_2.pdf")
    company_path = os.path.join("data", "company_data", "Company Data.pdf")
    
    logger.info(f"Testing with RFP: {rfp_path}")
    logger.info(f"Testing with Company Data: {company_path}")
    
    # Reset collections before test
    logger.info("Resetting ChromaDB collections...")
    reset_collections()
    
    try:
        # Initialize evaluator agent
        evaluator = EligibilityEvaluatorAgent()
        
        logger.info("Starting evaluation process...")
        result = evaluator.evaluate_eligibility(rfp_path, company_path)
        
        print("\nRFP Compliance Evaluation Results:")
        print("================================")
        
        if result["status"] == "error":
            print(f"Error: {result['message']}")
            return
        
        # Print core compliance status
        print("\nCore Compliance Status:")
        print("----------------------")
        print(result["evaluation"].split("Core Compliance Status:")[1].split("Required Submission Documents:")[0].strip())
        
        # Print submission requirements
        print("\nRequired Submission Documents:")
        print("-----------------------------")
        print(result["evaluation"].split("Required Submission Documents:")[1].split("Additional Desired Qualifications:")[0].strip())
        
        # Print additional qualifications
        print("\nAdditional Desired Qualifications:")
        print("----------------------------------")
        print(result["evaluation"].split("Additional Desired Qualifications:")[1].split("Overall Compliance Assessment:")[0].strip())
        
        # Print compliance assessment
        print("\nOverall Compliance Assessment:")
        print("-----------------------------")
        print(result["evaluation"].split("Overall Compliance Assessment:")[1].split("Required Actions:")[0].strip())
        
        # Print required actions
        print("\nRequired Actions:")
        print("---------------")
        print(result["evaluation"].split("Required Actions:")[1].strip())
        
        return result
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    test_evaluation()