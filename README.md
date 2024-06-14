# ChatFlex AI

This project presents ChatFlex AI, an advanced web application leveraging a custom-built Large Language Model (LLM) and Artificial Intelligence (AI). The LLM is meticulously designed to understand and generate human-like text with high accuracy and fluency, addressing the limitations of traditional applications that often lack adaptability and fail to meet unique user needs.

The development process involves leveraging an API for seamless integration with external systems, enabling the chatbot to interact with websites through web scraping capabilities. Features for handling PDF documents, including text extraction and summarization, are implemented to broaden the scope of the chatbot's utility. The system is architected to ensure scalability and optimized performance for large-scale usage, setting the stage for the future of AI-driven web applications. ChatFlex AI invites users to explore the limitless possibilities of intelligent communication, promising a new era in technological innovation and elevating online endeavors.

## Installation

### Prerequisites

- Python 3.x installed
- Git installed
- [Virtualenv](https://virtualenv.pypa.io/en/latest/) (optional but recommended)

### Clone the repository
git clone https://github.com/sindhuBasavaraj/ChatFlexAI.git
cd ChatFlexAI

### Setup Environment
Create and activate a virtual environment (optional but recommended):
pip install virtualenv

python -m venv env

.\env\Scripts\activate

### Install Dependencies
pip install -r requirements.txt

## Usage
To run the application, use the following command:

streamlit run app.py

This will start the Streamlit server and open the application in your default web browser.

## Environment Setup
To ensure the application runs correctly, follow these additional setup steps:

Create a .env file in the root directory of the project.

Add your Google API key to the .env file.

### API Keys Configuration
To run the project, you need to set up API keys.
Create a .env file in the root directory and add your API keys:


## .env file
GOOGLE_API_KEY=your_google_api_key_here

Make sure to replace your_google_api_key_here with your actual Google API key.


#### License
This project is licensed under the MIT License - see the LICENSE file for details.
