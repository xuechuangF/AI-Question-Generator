from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import os
port = int(os.environ.get('PORT', 5000))
import asyncio
import time
import json
import zipfile
import uuid
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import threading
from datetime import datetime
# 修改导入部分（第8-19行）
from core.generator import (
    EnhancedNoteToQuizGenerator,
    KnowledgePoint,
    Question,
    QuizFormatter,
    Config,  # 改为直接使用Config
    DocumentParser,
    TextChunker,
    KnowledgeExtractor,
    QuestionGenerator,
    KnowledgePointMerger,
    asdict
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 全局变量存储会话数据
sessions = {}
def allowed_file(filename):
    """检查文件类型是否允许"""
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'md', 'markdown'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Save file with proper extension handling
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 确保保留文件扩展名
            if '.' in filename:
                file_base, file_ext = os.path.splitext(filename)
                unique_filename = f"{timestamp}_{file_base}{file_ext}"
            else:
                # 如果文件名没有扩展名，尝试从MIME类型推断
                mime_to_ext = {
                    'application/pdf': '.pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                    'application/msword': '.doc',
                    'text/plain': '.txt',
                    'text/markdown': '.md'
                }
                file_ext = mime_to_ext.get(file.content_type, '')
                unique_filename = f"{timestamp}_{filename}{file_ext}"
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # 调试输出
            print(f"✅ 保存文件: {unique_filename}")
            print(f"✅ 完整路径: {file_path}")
            
            # Create configuration from form data
            config = Config()
            config.API_KEY = request.form.get('apiKey', '')
            config.QUALITY_LEVEL = request.form.get('qualityLevel', '中等')
            config.OUTPUT_DIR = os.path.join('outputs', session_id)
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            
            # Store session information
            sessions[session_id] = {
                'file_path': file_path,
                'config': config,
                'status': 'uploaded',
                'knowledge_points': None,
                'questions': None,
                'current_question': 0,
                'user_answers': {},
                'start_time': time.time(),
                'enable_review': request.form.get('enableReview') == 'true'
            }
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'filename': filename,
                'message': 'File uploaded successfully!'
            })
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
            
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large, please select a file smaller than 16MB'}), 400
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/process/<session_id>')
def process_file(session_id):
    """开始处理文件"""
    if session_id not in sessions:
        flash('会话不存在或已过期')
        return redirect(url_for('index'))
    
    session_data = sessions[session_id]
    
    # 检查是否已经在处理中或已完成
    if session_data['status'] in ['processing', 'completed']:
        # 如果已经在处理或完成，直接返回处理页面
        return render_template('processing.html', session_id=session_id)
    
    # 标记为处理中，防止重复处理
    session_data['status'] = 'processing'
    
    # 启动后台处理任务
    thread = threading.Thread(target=process_document_async, args=(session_id,))
    thread.daemon = True
    thread.start()
    
    return render_template('processing.html', session_id=session_id)

def process_document_async(session_id):
    """异步处理文档"""
    try:
        session_data = sessions[session_id]
        session_data['status'] = 'processing'
        
        # 添加调试信息
        file_path = session_data['file_path']
        print(f"🔍 处理文件路径: {file_path}")
        print(f"🔍 文件是否存在: {os.path.exists(file_path)}")
        print(f"🔍 文件扩展名: {os.path.splitext(file_path)[1]}")
        
        # 创建生成器
        generator = EnhancedNoteToQuizGenerator(session_data['config'])
        
        # 处理文档
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 不使用review参数，因为generator.process_document不接受这个参数
        knowledge_points, questions = loop.run_until_complete(
            generator.process_document(session_data['file_path'])
        )
        
        # 保存结果
        generator.save_results(knowledge_points, questions, session_data['config'].OUTPUT_DIR)
        
        # 将dataclass对象转换为字典以便存储
        session_data['knowledge_points'] = [asdict(kp) for kp in knowledge_points]
        session_data['questions'] = [asdict(q) for q in questions]
        session_data['status'] = 'completed'
        session_data['processing_time'] = time.time() - session_data['start_time']
        
    except Exception as e:
        sessions[session_id]['status'] = 'error'
        sessions[session_id]['error'] = str(e)
        import traceback
        traceback.print_exc()

@app.route('/status/<session_id>')
def get_status(session_id):
    """获取处理状态"""
    if session_id not in sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session_data = sessions[session_id]
    response = {
        'status': session_data['status']
    }
    
    if session_data['status'] == 'completed':
        response.update({
            'questions_count': len(session_data['questions']),
            'processing_time': session_data.get('processing_time', 0)
        })
    elif session_data['status'] == 'error':
        response['error'] = session_data.get('error', '未知错误')
    
    return jsonify(response)

@app.route('/review/<session_id>', methods=['GET', 'POST'])
def review_knowledge_points(session_id):
    """Knowledge point review page"""
    if session_id not in sessions:
        flash('Session not found or expired')
        return redirect(url_for('index'))
    
    session_data = sessions[session_id]
    if session_data['status'] != 'completed':
        flash('Document processing not completed')
        return redirect(url_for('process_file', session_id=session_id))
    
    if request.method == 'POST':
        # Handle knowledge point updates
        updated_kps = request.get_json().get('knowledge_points', [])
        kp_list = []
        for i, kp_data in enumerate(updated_kps, 1):
            kp = KnowledgePoint(
                id=i,
                title=kp_data['title'],
                summary=kp_data['summary'],
                context_ref=kp_data.get('context_ref', ''),
                key_formulas=kp_data.get('key_formulas', []),
                key_terms=kp_data.get('key_terms', []),
                difficulty_level=kp_data.get('difficulty_level', '基础'),
                knowledge_type=kp_data.get('knowledge_type', '概念定义')
            )
            kp_list.append(kp)
        session_data['knowledge_points'] = [asdict(kp) for kp in kp_list]
        return jsonify({'success': True, 'message': 'Knowledge points updated'})
    
    # Convert to KnowledgePoint objects for rendering
    kp_list = [KnowledgePoint(**kp) for kp in session_data['knowledge_points']]
    return render_template('review.html', 
                         session_id=session_id,
                         knowledge_points=kp_list)

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    """审核后生成题目"""
    data = request.get_json()
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    try:
        session_data = sessions[session_id]
        updated_kps = data.get('knowledge_points', [])
        
        # 转换为KnowledgePoint对象
        kp_objects = []
        for kp_data in updated_kps:
            kp = KnowledgePoint(
                id=kp_data['id'],
                title=kp_data['title'],
                summary=kp_data['summary'],
                context_ref=kp_data.get('context_ref', ''),
                key_formulas=kp_data.get('key_formulas', []),
                key_terms=kp_data.get('key_terms', []),
                difficulty_level=kp_data.get('difficulty_level', '基础'),
                knowledge_type=kp_data.get('knowledge_type', '概念定义')
            )
            kp_objects.append(kp)
        
        # 创建生成器并生成题目
        generator = EnhancedNoteToQuizGenerator(session_data['config'])
        
        # 使用异步方式生成题目
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        questions = loop.run_until_complete(generator.generator.generate_all(kp_objects))
        
        # 保存结果
        session_data['knowledge_points'] = [asdict(kp) for kp in kp_objects]
        session_data['questions'] = [asdict(q) for q in questions]
        
        # 保存到文件
        generator.save_results(kp_objects, questions, session_data['config'].OUTPUT_DIR)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'questions_count': len(questions)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/quiz/<session_id>')
def start_quiz(session_id):
    """开始做题"""
    if session_id not in sessions:
        flash('会话不存在或已过期')
        return redirect(url_for('index'))
    
    session_data = sessions[session_id]
    # 确保处理已完成
    if session_data['status'] != 'completed':
        flash('文档处理未完成')
        return redirect(url_for('process_file', session_id=session_id))
    
    questions = []
    for q_data in session_data['questions']:
        q = Question(**q_data)
        questions.append(q)
    
    # 重置做题状态
    session_data['current_question'] = 0
    session_data['user_answers'] = {}
    session_data['quiz_start_time'] = time.time()
    
    return render_template('quiz.html', 
                         session_id=session_id,
                         questions=questions,
                         current_question=0)

@app.route('/submit_answer/<session_id>', methods=['POST'])
def submit_answer(session_id):
    """Submit answer for a question"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = sessions[session_id]
    data = request.get_json()
    question_id = int(data.get('question_id'))
    answer = data.get('answer')  # 这是选项字母 A/B/C/D
    
    # 获取题目
    question = session_data['questions'][question_id]
    is_correct = answer == question['correct_answer']
    
    # 保存答案
    session_data['user_answers'][str(question_id)] = answer
    
    # 更新当前题目索引
    if question_id < len(session_data['questions']) - 1:
        session_data['current_question'] = question_id + 1
        is_last = False
    else:
        is_last = True
    
    return jsonify({
        'success': True,
        'is_correct': is_correct,
        'is_last': is_last,
        'explanation': question['explanation'],
        'correct_answer': question['correct_answer'],  # 返回正确答案字母
        'correct_answer_text': question['options'][question['correct_answer']]  # 返回正确答案内容
    })


@app.route('/results/<session_id>')
def show_results(session_id):
    """显示答题结果"""
    if session_id not in sessions:
        flash('会话不存在或已过期')
        return redirect(url_for('index'))
    
    session_data = sessions[session_id]
    questions = session_data['questions']
    user_answers = session_data['user_answers']
    
    #计算得分
    correct_count = 0
    results = []
    
    for i, question_dict in enumerate(questions):
        # 重建Question对象
        question = Question(**question_dict)
        user_answer = user_answers.get(str(i), '')
        is_correct = user_answer == question.correct_answer
        
        if is_correct:
            correct_count += 1
        
        results.append({
            'question': question.question,
            'options': question.options,
            'correct_answer': question.correct_answer,
            'user_answer': user_answer,
            'is_correct': is_correct,
            'explanation': question.explanation,
            'question_index': i
        })
    
    score = (correct_count / len(questions)) * 100 if questions else 0
    quiz_time = time.time() - session_data.get('quiz_start_time', 0)
    
    return render_template('results.html',
                         session_id=session_id,
                         results=results,
                         score=score,
                         correct_count=correct_count,
                         total_questions=len(questions),
                         quiz_time=quiz_time)

@app.route('/download/<session_id>/<format_type>')
def download_results(session_id, format_type):
    """下载结果文件"""
    if session_id not in sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session_data = sessions[session_id]
    if session_data['status'] != 'completed':
        return jsonify({'error': '处理未完成'}), 400
    
    try:
        # 恢复对象（新增）
        kp_list = [KnowledgePoint(**kp) for kp in session_data['knowledge_points']]
        q_list = [Question(**q) for q in session_data['questions']]
        
        # 创建生成器
        config = session_data['config']
        base_name = os.path.splitext(os.path.basename(session_data['file_path']))[0]
        output_dir = config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        if format_type == 'json':
            # 保存为JSON
            kp_data = [asdict(kp) for kp in kp_list]
            q_data = [asdict(q) for q in q_list]
            with open(f"{output_dir}/{base_name}_knowledge_points.json", 'w') as f:
                json.dump(kp_data, f, indent=2)
            with open(f"{output_dir}/{base_name}_questions.json", 'w') as f:
                json.dump(q_data, f, indent=2)
            
            # 创建压缩包
            zip_path = f"{output_dir}/{base_name}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(f"{output_dir}/{base_name}_knowledge_points.json", 
                          f"{base_name}_knowledge_points.json")
                zipf.write(f"{output_dir}/{base_name}_questions.json", 
                          f"{base_name}_questions.json")
            return send_file(zip_path, as_attachment=True)
            
        elif format_type == 'html':
            # 生成HTML
            html_content = QuizFormatter.to_html(q_list, kp_list)
            file_path = f"{output_dir}/{base_name}.html"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return send_file(file_path, as_attachment=True)
            
        elif format_type == 'markdown':
            # 生成Markdown
            md_content = QuizFormatter.to_markdown(q_list, kp_list)
            file_path = f"{output_dir}/{base_name}.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            return send_file(file_path, as_attachment=True)
            
        else:
            return jsonify({'error': '不支持的格式'}), 400
    
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/api/knowledge_points/<session_id>')
def get_knowledge_points(session_id):
    """获取知识点数据（用于AJAX）"""
    if session_id not in sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session_data = sessions[session_id]
    return jsonify({
        'knowledge_points': session_data.get('knowledge_points', []),
        'status': session_data['status']
    })

@app.route('/api/questions/<session_id>')
def get_questions(session_id):
    """获取题目数据（用于AJAX）"""
    if session_id not in sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session_data = sessions[session_id]
    return jsonify({
        'questions': session_data.get('questions', []),
        'current_question': session_data.get('current_question', 0)
    })

@app.errorhandler(413)
def too_large(e):
    flash('文件过大，请选择小于16MB的文件')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route('/api/session/<session_id>')
def get_session_info(session_id):
    """获取会话信息（用于判断是否需要审核）"""
    if session_id not in sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session_data = sessions[session_id]
    return jsonify({
        'status': session_data['status'],
        'enable_review': session_data.get('enable_review', False)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)