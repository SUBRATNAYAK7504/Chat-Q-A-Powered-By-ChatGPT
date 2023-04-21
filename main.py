from flask import Flask, request, jsonify, render_template
import openai

openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    # Process the chat message received from the client
    message = request.json['message']
    # Call ChatGPT API to get response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt= message,
        temperature=1,
        max_tokens=100,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        stop=None)

    response = response["choices"][0]["text"]
    #response = "Hi! I am a bot developed By Subrat"#chat_gpt_api(message)
    # Return the response to the client
    
    return jsonify({'message': response})

if __name__ == '__main__':
    app.run(debug=True)
