# Agentic - AI HTML Generator

An intelligent HTML webpage generator using CrewAI agents and Google Gemini. Simply describe your webpage in natural language, and the AI generates complete, styled HTML code for you.

## Features

- 🤖 **AI-Powered HTML Generation** - Generate complete webpages from descriptions
- 🎨 **Internal CSS Styling** - Automatic styling included in the generated HTML
- 💬 **Conversational Interface** - Chat-based webpage design
- 🚀 **Modern Stack** - Built with Streamlit, CrewAI, and Google Gemini 1.5 Pro
- 💾 **Chat History** - Maintains conversation history within session
- 📱 **Responsive Design** - Generated pages adapt to different screen sizes

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (get one [here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sujaanb/agentic.git
   cd agentic
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will open at `http://localhost:8501`

## Usage

1. Launch the application using the command above
2. In the chat input box, describe the webpage you want
3. The AI agent will generate complete HTML code
4. View the preview immediately and see the raw code
5. Continue the conversation to refine or create new pages

### Example Prompts

- "Create a beautiful landing page for a tech startup with a dark theme"
- "Make a contact form with blue and white colors"
- "Build a portfolio page for a photographer with image galleries"
- "Create a pricing table for a SaaS product"

## Project Structure

```
agentic/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── .devcontainer/        # Development container configuration
    └── devcontainer.json
```

## How It Works

The application uses a CrewAI agent with the following workflow:

1. **User Input** → Streamlit captures webpage description
2. **Agent Processing** → CrewAI's developer agent processes the request
3. **LLM Generation** → Gemini 1.5 Pro generates the HTML
4. **Validation** → Response is validated and formatted
5. **Display** → Both preview and code are shown to the user

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Your Google Gemini API key | ✅ Yes |

## Troubleshooting

### API Key Error
```
❌ GEMINI_API_KEY not found. Set it in .env or environment.
```
**Solution:** Make sure your `.env` file exists and contains a valid `GEMINI_API_KEY`

### Generation Errors
If the HTML generator throws an error:
1. Check your API key is valid
2. Verify you have sufficient API quota
3. Try a simpler prompt first
4. Check the logs for detailed error messages

### Streamlit Connection Issues
If Streamlit can't connect:
```bash
streamlit run app.py --logger.level=debug
```

## Development

### Using Dev Container

This project includes a VS Code Dev Container configuration for consistent development environment:

```bash
# Open in VS Code and select "Reopen in Container"
# Or use GitHub Codespaces
```

### Running Locally with Poetry (Alternative)

If you prefer Poetry:
```bash
poetry install
poetry run streamlit run app.py
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available under the MIT License.

## Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web app framework
- **[CrewAI](https://crewai.com/)** - Multi-agent orchestration
- **[Google Gemini](https://ai.google.dev/)** - LLM backend
- **[Python](https://www.python.org/)** - Programming language

## Future Enhancements

- [ ] Add support for multiple LLM providers
- [ ] Export generated HTML as files
- [ ] Add template library
- [ ] Support for JavaScript interactions
- [ ] Design suggestions based on best practices
- [ ] Multi-language support

## Contact & Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ by Sujaan Bhattacharyya**