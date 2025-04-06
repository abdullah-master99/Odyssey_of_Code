from agents.company_data_agent import CompanyDataAgent
import os
from datetime import datetime

def main():
    print("1. Testing Company Data Processing...")
    agent = CompanyDataAgent()
    
    # Process company data
    result = agent.execute_task(type('Task', (), {'name': 'process_company_data'}))
    print(f"Processing result: {result}")
    
    print("\n2. Testing Company Data Analysis...")
    # Test questions
    questions = [
        "What are the main services offered by the company?",
        "What is the company's experience in handling similar projects?",
        "What certifications or qualifications does the company have?",
        "What is the company's project management approach?",
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = agent.execute_task(
            type('Task', (), {'name': 'analyze_company_data'}),
            context={"query": question}
        )
        print(f"A: {result['analysis']}")
    
    print("\n3. Getting Company Data Statistics...")
    stats = agent.execute_task(type('Task', (), {'name': 'get_company_stats'}))
    print(f"Stats: {stats}")

if __name__ == "__main__":
    main()