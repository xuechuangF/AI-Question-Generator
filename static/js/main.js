class QuizGenerator {
    constructor() {
        this.currentStep = 1;
        this.sessionId = null;
        this.knowledgePoints = [];
        this.questions = [];
        this.currentQuestionIndex = 0;
        this.userAnswers = {};
        this.selectedFile = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.initMathJax();
    }

    initMathJax() {
        if (window.MathJax) {
            window.MathJax = {
                tex: {
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    processEscapes: true
                },
                options: {
                    skipHtmlTags: ['script', 'style', 'textarea']
                }
            };
        }
    }

    renderMath() {
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise().catch((err) => {
                console.warn('MathJax渲染错误:', err);
            });
        }
    }

    bindEvents() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        if (uploadArea && fileInput) {
            uploadArea.addEventListener('click', (e) => {
                if (e.target === uploadArea || e.target.parentElement === uploadArea) {
                    fileInput.click();
                }
            });
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    this.handleFileSelect(e.dataTransfer.files[0]);
                }
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileSelect(e.target.files[0]);
                }
            });
        }

        const nextBtn = document.getElementById('nextBtn');
        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.nextStep());
        }

        const generateBtn = document.getElementById('generateBtn');
        if (generateBtn) {
            generateBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.submitForm();
            });
        }
    }

    handleFileSelect(file) {
        const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'text/markdown'];
        
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(pdf|doc|docx|txt|md)$/i)) {
            alert('不支持的文件格式');
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            alert('文件大小不能超过16MB');
            return;
        }

        this.selectedFile = file;
        this.displayFileInfo(file);
        document.getElementById('nextBtn').disabled = false;
    }

    displayFileInfo(file) {
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(file.size);
        document.getElementById('fileInfo').style.display = 'block';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    nextStep() {
        if (!this.selectedFile) {
            alert('请先选择文件');
            return;
        }
        
        this.currentStep = 2;
        document.getElementById('uploadSection').style.display = 'none';
        document.getElementById('configSection').style.display = 'block';
        document.getElementById('backBtn').style.display = 'inline-block';
        document.getElementById('nextBtn').style.display = 'none';
        document.getElementById('generateBtn').style.display = 'inline-block';
        
        document.getElementById('step1').classList.remove('active');
        document.getElementById('step1').classList.add('completed');
        document.getElementById('step2').classList.add('active');
    }

    async submitForm() {
        const apiKey = document.getElementById('apiKey').value;
        if (!apiKey) {
            alert('请输入API密钥');
            return;
        }

        const formData = new FormData();
        formData.append('file', this.selectedFile);
        formData.append('apiKey', apiKey);
        formData.append('qualityLevel', document.getElementById('qualityLevel').value);
        formData.append('enableReview', document.getElementById('enableReview').checked);

        this.showLoading('正在上传文件...');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (result.success) {
                this.sessionId = result.session_id;
                this.processDocument();
            } else {
                throw new Error(result.error || '上传失败');
            }
        } catch (error) {
            this.hideLoading();
            alert('上传失败: ' + error.message);
        }
    }

    async processDocument() {
        this.showLoading('正在处理文档...');
        
        try {
            const response = await fetch(`/process/${this.sessionId}`);
            if (!response.ok) {
                throw new Error('处理请求失败');
            }
            
            // 开始轮询状态
            this.pollStatus();
        } catch (error) {
            this.hideLoading();
            alert('处理失败: ' + error.message);
        }
    }

    async pollStatus() {
        try {
            const response = await fetch(`/status/${this.sessionId}`);
            const result = await response.json();

            if (result.status === 'completed') {
                this.hideLoading();
                const enableReview = document.getElementById('enableReview').checked;
                if (enableReview) {
                    window.location.href = `/review/${this.sessionId}`;
                } else {
                    window.location.href = `/quiz/${this.sessionId}`;
                }
            } else if (result.status === 'error') {
                this.hideLoading();
                alert('处理失败: ' + (result.error || '未知错误'));
            } else {
                // 继续轮询
                setTimeout(() => this.pollStatus(), 2000);
            }
        } catch (error) {
            this.hideLoading();
            alert('获取状态失败: ' + error.message);
        }
    }

    showLoading(message) {
        document.getElementById('progressContainer').style.display = 'block';
        document.getElementById('progressText').textContent = message;
        document.getElementById('generateBtn').disabled = true;
    }

    hideLoading() {
        document.getElementById('progressContainer').style.display = 'none';
        document.getElementById('generateBtn').disabled = false;
    }
}

function clearFile() {
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('nextBtn').disabled = true;
    if (window.quizGenerator) {
        window.quizGenerator.selectedFile = null;
    }
}

function goBack() {
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('configSection').style.display = 'none';
    document.getElementById('backBtn').style.display = 'none';
    document.getElementById('nextBtn').style.display = 'inline-block';
    document.getElementById('generateBtn').style.display = 'none';
    document.getElementById('step2').classList.remove('active');
    document.getElementById('step1').classList.remove('completed');
    document.getElementById('step1').classList.add('active');
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    window.quizGenerator = new QuizGenerator();
});
