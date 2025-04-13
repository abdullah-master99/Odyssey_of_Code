from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from crewai import Crew, Task
from agents.subject_pdf_agent import SubjectPDFAgent
from agents.question_paper_agent import QuestionPaperAgent
from agents.paper_generator_agent import TeacherAssistantAgent
from utils import logger, feedback_analyzer, DIRS, result_tracker, reset_collections

# Reset collections on startup to use new model
logger.info("Resetting ChromaDB collections for new model...")
reset_collections()

app = Flask(__name__)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_type(file):
    """Validate that the file is a PDF"""
    if not file or not file.filename:
        return False, "No file provided"
        
    if not file.filename.lower().endswith('.pdf'):
        return False, "Only PDF files are allowed"
    
    return True, None

def create_crew():
    """Create and return a CrewAI crew with all agents and their coordinated workflow"""
    logger.info("Initializing agent workflow")
    
    try:
        # Initialize agents
        rfp_agent = SubjectPDFAgent()
        company_agent = QuestionPaperAgent()
        evaluator_agent = TeacherAssistantAgent()
        logger.info("All agents initialized successfully")
        
        # Define tasks with proper dependencies and expected outputs
        rfp_task = Task(
            description="Analyze subject or chapter PDF and extract relevant information for question paper generation",
            agent=rfp_agent,
            name="analyze_subject_pdf",
            expected_output="Dictionary containing extracted key topics, concepts, and relevant details for question paper creation"
        )
        
        company_task = Task(
            description="Analyze attached previous year/sample question paper to extract relevant information for question paper generation",
            agent=company_agent,
            name="analyze_company",
            expected_output="Dictionary containing extracted key topics, patterns, and insights from previous/sample question papers for question paper creation"
        )
        
        evaluation_task = Task(
            description="Generate relevant question paper based on extracted information",
            agent=evaluator_agent,
            name="generate_question_paper",
            expected_output="A well-structured question paper covering key topics and concepts extracted from the provided PDFs",
            context=[
            "Collect information dependencies from analyzed subject and sample question papers",
            "Generate a question paper that aligns with the extracted key topics and patterns",
            "Ensure the question paper includes a mix of difficulty levels and covers all critical areas",
            "Provide a comprehensive and balanced set of questions suitable for the intended audience"
            ],
            dependencies=[rfp_task, company_task]
        )
        
        # Create crew with defined workflow
        crew = Crew(
            agents=[rfp_agent, company_agent, evaluator_agent],
            tasks=[rfp_task, company_task, evaluation_task],
            verbose=True
        )
        
        logger.info("Agent workflow initialized successfully")
        return crew
        
    except Exception as e:
        logger.error(f"Error initializing agents: {str(e)}")
        raise

@app.route('/')
def home():
    """Render the main page"""
    return render_template('home.html')

@app.route('/upload/rfp', methods=['POST'])
def upload_rfp():
    """Handle RFP document upload"""
    try:
        logger.info("RFP upload request received")
        
        if 'rfp_file' not in request.files:
            logger.error("No rfp_file field in request")
            return jsonify({"status": "error", "error": "No file provided"}), 400
        
        file = request.files['rfp_file']
        logger.info(f"Received file: {file.filename}")
        
        # Validate file
        is_valid, error_message = validate_file_type(file)
        if not is_valid:
            logger.error(f"Invalid file: {error_message}")
            return jsonify({"status": "error", "error": error_message}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(DIRS['data']['rfps'], filename)
        logger.info(f"Saving file to: {file_path}")
        file.save(file_path)
        
        return jsonify({
            "status": "success",
            "message": "chapter file uploaded successfully",
            "filename": filename
        }), 200
        
    except Exception as e:
        logger.error(f"Error uploading RFP file: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/upload/company-data', methods=['POST'])
def upload_company_data():
    """Handle company data document upload"""
    try:
        logger.info("Company data upload request received")
        
        if 'company_file' not in request.files:
            logger.error("No company_file field in request")
            return jsonify({"status": "error", "error": "No file provided"}), 400
        
        file = request.files['company_file']
        logger.info(f"Received file: {file.filename}")
        
        # Validate file
        is_valid, error_message = validate_file_type(file)
        if not is_valid:
            logger.error(f"Invalid file: {error_message}")
            return jsonify({"status": "error", "error": error_message}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(DIRS['data']['company_data'], filename)
        logger.info(f"Saving file to: {file_path}")
        file.save(file_path)
        
        return jsonify({
            "status": "success",
            "message": "question paper file uploaded successfully",
            "filename": filename
        }), 200
    
    except Exception as e:
        logger.error(f"Error uploading company data file: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_eligibility():
    """Evaluate RFP eligibility using the CrewAI workflow"""
    try:
        data = request.get_json()
        if not data or 'rfp_file' not in data or 'company_file' not in data:
            return jsonify({"error": "Both RFP and company file names are required"}), 400
        
        rfp_path = os.path.join(DIRS['data']['rfps'], secure_filename(data['rfp_file']))
        company_path = os.path.join(DIRS['data']['company_data'], secure_filename(data['company_file']))
        
        if not (os.path.exists(rfp_path) and os.path.exists(company_path)):
            return jsonify({"error": "RFP or company file not found. Please upload files first."}), 404

        # Initialize evaluator agent
        evaluator = TeacherAssistantAgent()
        
        # Execute evaluation
        result = evaluator.generate_question_paper(rfp_path, company_path)
        
        if result["status"] == "error":
            return jsonify({
                "status": "error",
                "message": result["message"]
            }), 500

        # Save evaluation result
        evaluation_id = result_tracker.save_result(result)
        print(result)
        # Structure the response to match test evaluation format
        response = {
            "status": "success",
            "evaluation_id": evaluation_id,
            "question_paper": result["question_paper"],  # Send the full question paper text
            "sections": {
            "multiple_choice": result["question_paper"].get("multiple_choice", []),
            "short_answer": result["question_paper"].get("short_answer", []),
            "long_answer": result["question_paper"].get("long_answer", []),
            "case_studies": result["question_paper"].get("case_studies", []),
            "diagrams": result["question_paper"].get("diagrams", [])
            },
            "metadata": {
            "total_questions": result["question_paper"].get("total_questions", 0),
            "difficulty_distribution": result["question_paper"].get("difficulty_distribution", {}),
            "topics_covered": result["question_paper"].get("topics_covered", [])
            }
        }
        
        return jsonify(response), 200
            
    except Exception as e:
        logger.error(f"Error during eligibility evaluation: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for an RFP evaluation"""
    try:
        data = request.get_json()
        if not data or 'evaluation_id' not in data or 'feedback' not in data:
            return jsonify({"error": "Evaluation ID and feedback text are required"}), 400
        
        success = feedback_analyzer.save_feedback(
            data['evaluation_id'],
            data['feedback']
        )
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Feedback submitted successfully"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to save feedback"
            }), 500
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate')
def evaluate():
    """Render the evaluation page"""
    return render_template('index.html')

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)