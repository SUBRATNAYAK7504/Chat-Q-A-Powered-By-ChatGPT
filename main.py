from flask import Flask, request, jsonify, render_template

import io
import PyPDF2

from answer_generator import *



VERBOSE_MODE = True
train_df = ""

app = Flask(__name__)

PDF_FLAG = False
URL_FLAG = False

@app.route('/')
def index():
    if VERBOSE_MODE: print("Rendering Home Page")
    return render_template('chat.html')

@app.route('/api/pdfdata', methods = ["POST"])
def pdf_test():
    global PDF_FLAG 
    PDF_FLAG = True
    global train_df
    if VERBOSE_MODE: print("PDF Reader API has called")
    pdf_files = request.files.getlist("file")
    full_text = []
    if VERBOSE_MODE: print("\tReading text from pdf pages")
    for f in pdf_files:
        if VERBOSE_MODE: print(f"\t\t{f} pdf file is being read..")
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(f.read()))
        # Get the total number of pages in the PDF file
        num_pages = len(pdf_reader.pages)

        # Loop through all the pages in the PDF file
        full_text = []
        
        for page_num in range(4, num_pages):

            if VERBOSE_MODE and page_num%10 == 0: print(f"\t\t\tReading {page_num} page out of {num_pages} pages")
            # Get the page object for the current page
            page = pdf_reader.pages[page_num]
            
            # Extract the text from the page and add it to the string
            text = page.extract_text()
            full_text.append((page_num+1, text, f))

    if VERBOSE_MODE: print("\tPDF read successfully")

    if VERBOSE_MODE: print("Training Data is being generated")
    train_df = preprocess_data(full_text)
    if VERBOSE_MODE: print("Training Data generated successfully")
    #print(text)

    return jsonify({"message": "Successfully received files in backend"})

@app.route('/api/chat', methods=['POST'])
def chat():
    if VERBOSE_MODE: print("Getting Response from ChatGPT")
    # Process the chat message received from the client
    message = request.json['message']
    if PDF_FLAG == True:
        response, source = answer_question(train_df, question = message, doc_type="pdf")
    
        print(source)

    if VERBOSE_MODE: print("Got the response from ChatGPT")
    return jsonify({'message': response, 'source': source})

if __name__ == '__main__':
    app.run(debug=True)
