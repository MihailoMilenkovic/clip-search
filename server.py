from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.H1('Flask-Dash Image and Text Application'),
    
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H2('Image Upload'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    
    dcc.Link('Go to Text Input', href='/text', style={'margin': '10px'})
])

text_page = html.Div([
    html.H2('Text Input'),
    dcc.Textarea(
        id='text-input',
        placeholder='Enter text...',
        style={'width': '100%', 'height': '100px'}
    ),
    html.Button('Generate Image', id='generate-image-button', n_clicks=0),
    html.Div(id='output-text-input'),
    
    dcc.Link('Go to Image Upload', href='/', style={'margin': '10px'})
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return index_page
    elif pathname == '/text':
        return text_page
    else:
        return '404 - Page not found'

@app.callback(
    Output('output-image-upload', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def process_uploaded_image(content, filename):
  #TODO: customize with searching CLIP embeddings
    if content is not None:
        image = Image.open(io.BytesIO(content.encode('utf-8')))
        
        # Process the image and generate a textual response
        # Replace this with your own image processing code
        response_text = f'Textual response for image: {filename}'
        
        return html.Div([
            html.H3('Uploaded Image:'),
            html.Img(src=content, style={'height': '300px'}),
            html.H4('Textual Response:'),
            html.Div(response_text)
        ])

@app.callback(
    Output('output-text-input', 'children'),
    [Input('generate-image-button', 'n_clicks')],
    [State('text-input', 'value')]
)
def generate_image(n_clicks, text):
  #TODO: customize with searching CLIP embeddings
    if n_clicks > 0:
        # Process the text and generate an image response
        # Replace this with your own image generation code
        response_image = Image.new('RGB', (200, 200), (255, 255, 255))
        image_io = io.BytesIO()
        response_image.save(image_io, 'PNG')
        image_io.seek(0)
        
        return html.Div([
            html.H3('Text Input:'),
            html.Div(text),
            html.H4('Generated Image:'),
            html.Img(src='data:image/png;base64,{}'.format(image_io.getvalue()), style={'height': '300px'})
        ])

if __name__ == '__main__':
    app.run_server(debug=True)