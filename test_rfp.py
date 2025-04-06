from agents.rfp_extractor_agent import RFPAgent
import os

def test_rfp_processing_and_qa():
    try:
        # Initialize agent
        agent = RFPAgent()
        
        # Test RFP processing
        print("\n1. Testing RFP Processing...")
        process_task = type('Task', (), {'name': 'process_rfp'})()
        result = agent.execute_task(process_task)
        
        if "error" in result:
            print(f"Error processing RFP: {result['error']}")
            return False
            
        print(f"Successfully processed RFP: {result['file']}")
        print(f"Number of chunks processed: {result['chunks_processed']}")
        
        # Test question answering
        print("\n2. Testing Question Answering...")
        test_questions = [
            "What is the scope of work in this RFP?",
            "What are the eligibility criteria?",
            "What is the submission deadline?",
            "What is the estimated budget?"
        ]
        
        qa_task = type('Task', (), {'name': 'answer_question'})()
        
        for question in test_questions:
            print(f"\nQ: {question}")
            result = agent.execute_task(qa_task, context={"question": question})
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"A: {result['answer']}")
                
        return True

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
    success = test_rfp_processing_and_qa()
    exit(0 if success else 1)