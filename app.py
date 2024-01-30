from flask import Flask, request,redirect,url_for,render_template
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
from yolov5.detect import run
import yaml
import random,string
from os import listdir
from flask_bootstrap import Bootstrap5


app = Flask(__name__)
bootstrap = Bootstrap5(app)
def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        return config

config = load_config()
app.config.update(config)

# @app.context_processor
# def utility_processor():
def get_yolo_filepath(filename,version):
        
    return os.path.join(app.config['process_folder'],version,'result', filename)


app.jinja_env.globals.update(get_yolo_filepath=get_yolo_filepath)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config.get('allowed_image_extensions')+app.config.get('allowed_video_extensions')
           


def is_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config.get('allowed_image_extensions')
           
def is_video(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config.get('allowed_video_extensions')





@app.route('/result-separate-image', methods=['GET'])
def result_separate_image():
    return render_template("show_image_results.html", file_v5=request.args.get("file_v5"),file_v8=request.args.get("file_v8"),filename=request.args.get("filename"))


@app.route('/result-separate-video', methods=['GET'])
def result_separate_video():
    return render_template("show_video_results.html", file_v5=request.args.get("file_v5"),file_v8=request.args.get("file_v8"),filename=request.args.get("filename"))
           
           

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            
            extension = file.filename.split('.')[-1]
            filename_prefix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            file.filename = filename_prefix+"."+extension
            
            filename = secure_filename(file.filename)
            
            uploaded_filepath = os.path.join(app.config["static_folder"],app.config['upload_folder'], filename)
            file.save(uploaded_filepath)
            
            v5_filepath = os.path.join(app.config["static_folder"],app.config['process_folder'],"v5")
            
                  
            run(weights=app.config["v5_weights_folder"],conf_thres=0.25,
                imgsz=(640,640),
                source=uploaded_filepath, 
                project=v5_filepath,
                name="result",
                exist_ok=True
                )
            
            v8_filepath =os.path.join(app.config["static_folder"],app.config['process_folder'],"v8")
            
            model = YOLO(app.config["v8_weights_folder"],)
            result = model.predict(source=uploaded_filepath,
                project=v8_filepath,
                name="result",
                exist_ok=True,
                save=True
                )
            
            if is_image(filename):
            
            
                return redirect(
                    url_for("result_separate_image",file_v5= os.path.join(app.config['process_folder'],"v5","result", filename),
                            file_v8= os.path.join(app.config['process_folder'],"v8","result", filename),
                            filename= filename
                            )
                    )
            else:
            
                return redirect(
                    url_for("result_separate_video",file_v5= os.path.join(app.config['process_folder'],"v5","result", filename),
                            file_v8= os.path.join(app.config['process_folder'],"v8","result", filename),
                            filename= filename
                            )
                    )
                
    return  render_template("upload.html")



@app.route('/results', methods=['GET'])
def results():
    upload_files = listdir(os.path.join(app.config["static_folder"],app.config['upload_folder']))
    
    images =[ upload_file for upload_file in upload_files if is_image(upload_file) ]
    videos =[upload_file for upload_file in upload_files if is_video(upload_file)]
    
    
    return render_template("results.html", 
                           images=images,
                           videos =videos,
                           destination_folder=app.config['upload_folder']
                           )

if __name__ == "__main__":
    app.run(debug=True) 

