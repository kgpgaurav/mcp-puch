import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (now smart!) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# Image inputs and sending images

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Email Tools ---

EMAIL_SUMMARIZER_DESCRIPTION = RichToolDescription(
    description="Summarize emails and answer questions about email content.",
    use_when="Use this tool when user provides email content and wants summary or has questions about it.",
    side_effects="Analyzes email content and provides structured summary or answers specific questions.",
)

@mcp.tool(description=EMAIL_SUMMARIZER_DESCRIPTION.model_dump_json())
async def email_analyzer(
    email_content: Annotated[str, Field(description="The complete email content including headers, body, and any metadata")],
    analysis_type: Annotated[str, Field(description="Type of analysis: 'summarize', 'action_items', 'sentiment', 'reply_suggestions', 'question'")] = "summarize",
    specific_question: Annotated[str | None, Field(description="Specific question about the email (only used when analysis_type is 'question')")] = None,
) -> str:
    """
    Analyze email content and provide summaries, action items, sentiment analysis, or answer questions.
    """
    if not email_content or len(email_content.strip()) < 10:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide valid email content."))

    # Extract basic email components
    lines = email_content.strip().split('\n')
    
    # Try to identify email structure
    subject = ""
    sender = ""
    body_start = 0
    
    for i, line in enumerate(lines[:20]):  # Check first 20 lines for headers
        line_lower = line.lower().strip()
        if line_lower.startswith('subject:'):
            subject = line[8:].strip()
        elif line_lower.startswith('from:'):
            sender = line[5:].strip()
        elif line.strip() == "" and i > 0:  # Empty line usually marks start of body
            body_start = i + 1
            break
    
    # Extract body content
    body_lines = lines[body_start:] if body_start > 0 else lines
    body = '\n'.join(body_lines).strip()
    
    # Remove excessive whitespace and clean up
    body = ' '.join(body.split())
    
    if analysis_type == "summarize":
        return f"""📧 **Email Summary**

**Subject**: {subject or "No subject found"}
**From**: {sender or "Sender not identified"}

**📋 Key Points:**
{_extract_key_points(body)}

**💡 Main Message:**
{_extract_main_message(body)}

**📊 Email Stats:**
- Length: {len(body)} characters
- Word count: {len(body.split())} words
- Estimated reading time: {max(1, len(body.split()) // 200)} minute(s)
"""

    elif analysis_type == "action_items":
        return f"""📧 **Action Items Analysis**

**Subject**: {subject or "No subject found"}

**✅ Action Items Identified:**
{_extract_action_items(body)}

**⏰ Time-Sensitive Items:**
{_extract_time_sensitive_items(body)}

**👥 People to Follow Up With:**
{_extract_people_to_contact(body)}
"""

    elif analysis_type == "sentiment":
        return f"""📧 **Email Sentiment Analysis**

**Subject**: {subject or "No subject found"}

**😊 Overall Tone**: {_analyze_sentiment(body)}

**🎯 Intent**: {_analyze_intent(body)}

**🚨 Urgency Level**: {_analyze_urgency(body)}

**💼 Formality Level**: {_analyze_formality(body)}
"""

    elif analysis_type == "reply_suggestions":
        return f"""📧 **Reply Suggestions**

**Subject**: {subject or "No subject found"}

**💬 Suggested Responses:**

**Quick Replies:**
{_suggest_quick_replies(body)}

**Detailed Response Template:**
{_suggest_detailed_response(body)}

**📝 Key Points to Address:**
{_extract_points_to_address(body)}
"""

    elif analysis_type == "question":
        if not specific_question:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide a specific question when using 'question' analysis type."))
        
        return f"""📧 **Question Analysis**

**Your Question**: {specific_question}

**📍 Answer Based on Email Content:**
{_answer_question_about_email(body, specific_question)}

**📋 Relevant Email Excerpts:**
{_find_relevant_excerpts(body, specific_question)}
"""

    else:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid analysis_type. Use: 'summarize', 'action_items', 'sentiment', 'reply_suggestions', or 'question'"))

def _extract_key_points(body: str) -> str:
    """Extract key points from email body"""
    # Simple keyword-based extraction
    sentences = body.split('.')
    key_sentences = []
    
    keywords = ['important', 'urgent', 'deadline', 'meeting', 'please', 'required', 'need', 'must', 'asap', 'immediately']
    
    for sentence in sentences[:10]:  # Limit to first 10 sentences
        sentence = sentence.strip()
        if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in keywords):
            key_sentences.append(f"• {sentence.strip()}")
    
    if not key_sentences:
        # If no keyword matches, take first few meaningful sentences
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if len(sentence) > 30:
                key_sentences.append(f"• {sentence}")
    
    return '\n'.join(key_sentences[:5]) or "• No specific key points identified"

def _extract_main_message(body: str) -> str:
    """Extract main message from email"""
    # Take first substantial paragraph
    paragraphs = [p.strip() for p in body.split('\n\n') if len(p.strip()) > 50]
    
    if paragraphs:
        main_msg = paragraphs[0]
        if len(main_msg) > 200:
            main_msg = main_msg[:200] + "..."
        return main_msg
    
    # Fallback to first 150 characters
    return body[:150] + "..." if len(body) > 150 else body

def _extract_action_items(body: str) -> str:
    """Extract action items from email"""
    action_words = ['please', 'need you to', 'can you', 'would you', 'could you', 'action required', 'next steps', 'todo', 'follow up']
    
    sentences = body.split('.')
    actions = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if any(word in sentence_lower for word in action_words):
            actions.append(f"• {sentence.strip()}")
    
    return '\n'.join(actions[:5]) or "• No specific action items identified"

def _extract_time_sensitive_items(body: str) -> str:
    """Extract time-sensitive information"""
    time_keywords = ['deadline', 'due', 'urgent', 'asap', 'immediately', 'today', 'tomorrow', 'this week', 'by end of']
    
    sentences = body.split('.')
    time_items = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if any(keyword in sentence_lower for keyword in time_keywords):
            time_items.append(f"⏰ {sentence.strip()}")
    
    return '\n'.join(time_items[:3]) or "⏰ No urgent deadlines identified"

def _extract_people_to_contact(body: str) -> str:
    """Extract people mentioned for follow-up"""
    # Simple email extraction
    import re
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', body)
    
    people = []
    if emails:
        for email in emails[:3]:
            people.append(f"👤 {email}")
    
    # Look for name patterns
    name_patterns = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', body)
    for name in name_patterns[:2]:
        people.append(f"👤 {name}")
    
    return '\n'.join(people[:5]) or "👤 No specific contacts identified"

def _analyze_sentiment(body: str) -> str:
    """Simple sentiment analysis"""
    positive_words = ['thank', 'great', 'excellent', 'good', 'appreciate', 'wonderful', 'fantastic']
    negative_words = ['problem', 'issue', 'concern', 'urgent', 'wrong', 'error', 'disappointed']
    
    body_lower = body.lower()
    positive_count = sum(1 for word in positive_words if word in body_lower)
    negative_count = sum(1 for word in negative_words if word in body_lower)
    
    if positive_count > negative_count:
        return "Positive/Friendly"
    elif negative_count > positive_count:
        return "Concerned/Negative"
    else:
        return "Neutral/Professional"

def _analyze_intent(body: str) -> str:
    """Analyze email intent"""
    body_lower = body.lower()
    
    if any(word in body_lower for word in ['meeting', 'schedule', 'calendar']):
        return "Meeting/Scheduling"
    elif any(word in body_lower for word in ['question', 'help', 'clarification']):
        return "Information Request"
    elif any(word in body_lower for word in ['update', 'status', 'progress']):
        return "Status Update"
    elif any(word in body_lower for word in ['please', 'need', 'request']):
        return "Action Request"
    else:
        return "General Communication"

def _analyze_urgency(body: str) -> str:
    """Analyze urgency level"""
    urgent_words = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
    
    if any(word in body.lower() for word in urgent_words):
        return "High - Immediate attention needed"
    elif any(word in body.lower() for word in ['soon', 'quick', 'deadline']):
        return "Medium - Timely response needed"
    else:
        return "Low - No rush"

def _analyze_formality(body: str) -> str:
    """Analyze formality level"""
    formal_indicators = ['dear', 'sincerely', 'regards', 'respectfully']
    informal_indicators = ['hey', 'hi', 'thanks', 'cheers']
    
    body_lower = body.lower()
    
    if any(word in body_lower for word in formal_indicators):
        return "Formal"
    elif any(word in body_lower for word in informal_indicators):
        return "Informal/Casual"
    else:
        return "Semi-formal"

def _suggest_quick_replies(body: str) -> str:
    """Suggest quick reply options"""
    body_lower = body.lower()
    
    replies = []
    
    if 'meeting' in body_lower:
        replies.append("• 'I'm available for the meeting. Please share the agenda.'")
        replies.append("• 'Let me check my calendar and get back to you.'")
    
    if any(word in body_lower for word in ['question', 'help']):
        replies.append("• 'Happy to help! Let me look into this and respond shortly.'")
        replies.append("• 'Thanks for reaching out. I'll get back to you with details.'")
    
    if 'thank' in body_lower:
        replies.append("• 'You're welcome! Let me know if you need anything else.'")
    
    if not replies:
        replies = [
            "• 'Thank you for your email. I'll review and respond accordingly.'",
            "• 'Received your message. I'll get back to you soon.'",
            "• 'Thanks for the update. Let me know if you need anything.'"
        ]
    
    return '\n'.join(replies[:3])

def _suggest_detailed_response(body: str) -> str:
    """Suggest detailed response structure"""
    return """**Suggested Response Structure:**

1. **Acknowledgment**: Thank them for their email
2. **Address Main Points**: Respond to key questions/requests
3. **Action Items**: Clarify what you'll do next
4. **Timeline**: When they can expect follow-up
5. **Closing**: Professional sign-off"""

def _extract_points_to_address(body: str) -> str:
    """Extract key points that need to be addressed in reply"""
    # Look for question marks and action words
    sentences = body.split('.')
    points = []
    
    for sentence in sentences:
        if '?' in sentence or any(word in sentence.lower() for word in ['please', 'need', 'can you']):
            points.append(f"• {sentence.strip()}")
    
    return '\n'.join(points[:4]) or "• Acknowledge receipt and provide requested information"

def _answer_question_about_email(body: str, question: str) -> str:
    """Answer specific question about email content"""
    question_lower = question.lower()
    body_lower = body.lower()
    
    # Simple keyword matching approach
    if 'who' in question_lower:
        # Look for names/emails
        import re
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', body)
        names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', body)
        
        if emails or names:
            return f"People mentioned: {', '.join(emails + names)}"
    
    elif 'when' in question_lower:
        # Look for dates/times
        import re
        dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}-\d{1,2}-\d{2,4}\b', body)
        times = re.findall(r'\b\d{1,2}:\d{2}\b', body)
        
        if dates or times:
            return f"Times/dates mentioned: {', '.join(dates + times)}"
    
    elif 'what' in question_lower:
        # Extract main topic
        return f"Main topic appears to be: {_extract_main_message(body)[:100]}..."
    
    # Fallback: find sentences containing question keywords
    question_words = question_lower.split()
    relevant_sentences = []
    
    for sentence in body.split('.'):
        if any(word in sentence.lower() for word in question_words):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        return f"Based on the email content: {' '.join(relevant_sentences[:2])}"
    
    return "I couldn't find specific information related to your question in the email content."

def _find_relevant_excerpts(body: str, question: str) -> str:
    """Find relevant excerpts from email based on question"""
    question_words = question.lower().split()
    sentences = body.split('.')
    
    relevant = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and any(word in sentence.lower() for word in question_words):
            relevant.append(f"• \"{sentence}\"")
    
    return '\n'.join(relevant[:3]) or "• No directly relevant excerpts found"

# --- Run MCP Server ---
async def main():
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting MCP server on http://0.0.0.0:{port}")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=port)

if __name__ == "__main__":
    asyncio.run(main())
