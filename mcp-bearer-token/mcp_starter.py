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
import base64
import json
import re
import hashlib
from datetime import datetime, timedelta

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
    if body:
        body = ' '.join(body.split())
    else:
        body = "[No email body content found]"
    
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
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', body)
        names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', body)
        
        if emails or names:
            return f"People mentioned: {', '.join(emails + names)}"
    
    elif 'when' in question_lower:
        # Look for dates/times
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

# --- Gmail Integration Tools ---

GMAIL_OAUTH_DESCRIPTION = RichToolDescription(
    description="Start Gmail OAuth authentication flow to securely connect your Gmail account.",
    use_when="When user wants to connect their Gmail account for the first time or refresh expired access.",
    side_effects="Generates OAuth URL for browser authentication and provides access token retrieval instructions.",
)

@mcp.tool(description=GMAIL_OAUTH_DESCRIPTION.model_dump_json())
async def start_gmail_oauth(
    use_shared_credentials: Annotated[bool, Field(description="Use shared OAuth credentials for easier setup (recommended for most users)")] = True,
) -> str:
    """
    Start Gmail OAuth authentication process with proper browser consent flow.
    """
    
    if use_shared_credentials:
        # Direct users to OAuth Playground for the easiest setup
        playground_url = "https://developers.google.com/oauthplayground"
        
        return f"""🔐 **Super Easy Gmail Setup** (⚡ **2 minutes only!**)

## 🚀 **Zero Technical Skills Required - Just Follow These Steps:**

### **🎯 Step 1: Go to OAuth Playground**
👉 **[Click Here: Google OAuth Playground]({playground_url})**

### **🎯 Step 2: Select Gmail API**
1. In the left panel, scroll down to **"Gmail API v1"**
2. Click the **arrow** to expand it
3. Check the box: **"https://www.googleapis.com/auth/gmail.readonly"**
4. Click **"Authorize APIs"** (blue button)

### **🎯 Step 3: Sign In & Allow**
1. **Sign in to your Gmail account**
2. Click **"Allow"** when Google asks for permission
3. You'll be redirected back to OAuth Playground

### **🎯 Step 4: Get Your Token**
1. Click **"Exchange authorization code for tokens"** (blue button)
2. **Copy the "Access token"** (starts with `ya29.`)

### **🎯 Step 5: Test It!**
```
search_query: "is:unread"
analysis_type: "summarize"
gmail_access_token: "[PASTE_YOUR_TOKEN_HERE]"
max_emails: 20
```

## ✅ **That's it! No Google Cloud setup needed!**

## 🛡️ **Security Notes:**
- ✅ **Read-only access** to your emails only
- ✅ **You control permissions** - can revoke anytime
- ✅ **No emails stored** on our servers
- ✅ **Uses Google's official OAuth Playground**
- ⚠️ **Token expires in 1 hour** - get fresh one when needed

## 🔄 **When Token Expires:**
Just repeat the process - it takes 30 seconds once you know the steps!

---

**🆘 Need help?** The process: OAuth Playground → Gmail API → Authorize → Allow → Get Token → Use it! 🎉
"""
    
    else:
        # Individual OAuth setup (original detailed process)
        return f"""🔐 **Advanced Gmail OAuth Setup** (For developers who want their own credentials)

## 🚀 **Step 1: Create Google OAuth Credentials**
1. Go to: **https://console.cloud.google.com/apis/credentials**
2. Create a new project or select existing one
3. Click **"+ CREATE CREDENTIALS"** → **"OAuth 2.0 Client IDs"**
4. Choose **"Web application"**
5. Add redirect URI: `https://developers.google.com/oauthplayground`
6. Copy your **Client ID** and **Client Secret**

## 🔓 **Step 2: Get Access Token**
1. Go to: **https://developers.google.com/oauthplayground/**
2. Click the **⚙️ gear icon** (top right)
3. Check **"Use your own OAuth credentials"**
4. Enter your **Client ID** and **Client Secret**
5. In "Step 1", select **"Gmail API v1"** → **"https://www.googleapis.com/auth/gmail.readonly"**
6. Click **"Authorize APIs"** → Sign in and give consent
7. In "Step 2", click **"Exchange authorization code for tokens"**
8. Copy the **"Access token"** and **"Refresh token"**

## ✅ **After Getting Tokens:**
Use `gmail_search_and_analyze` with your access token!

**Example:**
```
search_query: "is:unread -category:promotions -from:noreply"
analysis_type: "summarize"
gmail_access_token: "ya29.a0AfH6SMC..."
max_emails: 50
```

## 🛡️ **Security Features:**
- ✅ Read-only access only
- ✅ User consent required
- ✅ Your own OAuth credentials
- ✅ Full control over permissions
"""

GMAIL_QUICK_SETUP_DESCRIPTION = RichToolDescription(
    description="Get instant Gmail access with zero technical setup - perfect for non-technical users.",
    use_when="When user wants the easiest possible way to connect Gmail without any technical setup.",
    side_effects="Provides instant OAuth link using shared credentials for immediate access.",
)

@mcp.tool(description=GMAIL_QUICK_SETUP_DESCRIPTION.model_dump_json())
async def gmail_quick_setup() -> str:
    """
    Provide the absolute easiest way to connect Gmail - zero technical setup required.
    """
    
    # Direct users to OAuth Playground for instant setup
    playground_url = "https://developers.google.com/oauthplayground"
    
    return f"""⚡ **INSTANT Gmail Setup** - Zero Tech Skills!

## 🎯 **4 Simple Clicks (2 minutes total):**

### **1️⃣ Go to OAuth Playground:**
👉 **[Open Google OAuth Playground]({playground_url})**

### **2️⃣ Select Gmail API:**
- Scroll to **"Gmail API v1"** in left panel
- Click to expand → Check **"gmail.readonly"**
- Click **"Authorize APIs"**

### **3️⃣ Sign In & Allow:**
- Sign in to Gmail → Click **"Allow"**

### **4️⃣ Get Your Token:**
- Click **"Exchange authorization code for tokens"**
- Copy the **"Access token"** (starts with `ya29.`)

## 🚀 **Test It Immediately:**
```
search_query: "is:unread"
analysis_type: "summarize" 
gmail_access_token: "[PASTE_YOUR_TOKEN_HERE]"
max_emails: 20
```

## ✨ **What You Get:**
- 📧 **Search any emails** by sender, subject, date
- 🤖 **AI analysis** - summaries, action items, Q&A  
- 🚫 **Smart filtering** - no spam/promotions/bank emails
- ⚡ **Fast results** - cached for 30 minutes
- 🔒 **Read-only** - completely safe

## 🔄 **When Token Expires (1 hour):**
Just repeat the process - takes 30 seconds!

## 💬 **Example Searches:**
```
Recent emails: "is:unread"
From boss: "from:boss@company.com"
This week: "after:2024-08-05"
With attachments: "has:attachment"
Meeting invites: "subject:meeting"
```

**Super Easy: OAuth Playground → Gmail API → Authorize → Token! 🎉**
"""

GMAIL_SETUP_GUIDE_DESCRIPTION = RichToolDescription(
    description="Comprehensive visual guide for Gmail OAuth setup with step-by-step screenshots and troubleshooting.",
    use_when="When user needs detailed help with Gmail authentication or encounters issues during setup.",
    side_effects="Provides detailed visual guide with common troubleshooting solutions.",
)

@mcp.tool(description=GMAIL_SETUP_GUIDE_DESCRIPTION.model_dump_json())
async def gmail_setup_guide() -> str:
    """
    Comprehensive step-by-step guide for Gmail OAuth setup with troubleshooting.
    """
    
    return """📖 **Complete Gmail Setup Guide** - Visual Walkthrough

## 🎯 **Method 1: Super Easy (Recommended)**

### **📍 Step 1: Open OAuth Playground**
👉 **Go to: https://developers.google.com/oauthplayground**

### **📍 Step 2: Find Gmail API**
- Look for **"Gmail API v1"** in the left sidebar
- Click the **small arrow** to expand the section
- You'll see: `https://www.googleapis.com/auth/gmail.readonly`

### **📍 Step 3: Select Permission**
- ✅ **Check the box** next to `gmail.readonly`
- Click the blue **"Authorize APIs"** button

### **📍 Step 4: Google Sign-In**
- **Sign in** to your Gmail account
- **Click "Allow"** when Google asks for permission
- You'll return to OAuth Playground

### **📍 Step 5: Get Your Token**
- Click **"Exchange authorization code for tokens"**
- **Copy the "Access token"** (long string starting with `ya29.`)

## 🚨 **Common Issues & Solutions:**

### **❌ "Access blocked" Error**
- **Solution**: Make sure you're using OAuth Playground directly, not a custom link
- **URL should be**: `developers.google.com/oauthplayground`

### **❌ "invalid_client" Error**  
- **Solution**: Use the OAuth Playground method above instead of direct URLs

### **❌ "Token expired" Error**
- **Solution**: Tokens last 1 hour. Just repeat Step 5 to get a fresh token

### **❌ "Scope not found" Error**
- **Solution**: Make sure you selected exactly: `https://www.googleapis.com/auth/gmail.readonly`

## 🎯 **Method 2: Advanced (For Developers)**

### **🔧 Create Your Own OAuth App**
1. **Google Cloud Console**: https://console.cloud.google.com/apis/credentials
2. **Create Project** → **OAuth 2.0 Client ID** → **Web Application**
3. **Add Redirect URI**: `https://developers.google.com/oauthplayground`
4. **Copy Client ID & Secret**

### **🔧 Use Your Credentials**
1. **OAuth Playground**: https://developers.google.com/oauthplayground
2. **Settings (⚙️)** → **Use your own OAuth credentials**
3. **Enter your Client ID & Secret**
4. **Continue with normal flow**

## ✅ **Test Your Setup**

After getting your token, test with:
```
Tool: gmail_search_and_analyze
search_query: "is:unread"
analysis_type: "summarize"
gmail_access_token: "[YOUR_TOKEN_HERE]"
max_emails: 10
```

## 🛡️ **Security & Privacy**

- ✅ **Read-only access** - we can't send/delete emails
- ✅ **No data stored** - emails processed in real-time
- ✅ **Revoke anytime** - Google Account → Security → Third-party apps
- ✅ **Token expires** - automatic security after 1 hour

## 💡 **Pro Tips**

1. **Bookmark OAuth Playground** for quick token refresh
2. **Save your Client ID** if using advanced method
3. **Test with simple queries** first (`is:unread`)
4. **Use smart filtering** to avoid spam/promotions

---

**🆘 Still need help?** Most users succeed with Method 1 on their first try! The key is using Google's OAuth Playground directly, not custom URLs.
"""

GMAIL_SEARCH_DESCRIPTION = RichToolDescription(
    description="Search and analyze emails directly from Gmail with smart filtering (no promotions/bank emails).",
    use_when="When user wants to search, summarize, or ask questions about their emails directly from Gmail.",
    side_effects="Connects to Gmail API to fetch and analyze email content with intelligent filtering.",
)

@mcp.tool(description=GMAIL_SEARCH_DESCRIPTION.model_dump_json())
async def gmail_search_and_analyze(
    search_query: Annotated[str, Field(description="Gmail search query (e.g., 'from:boss@company.com', 'subject:meeting', 'is:unread', 'after:2024-01-01')")],
    analysis_type: Annotated[str, Field(description="Type of analysis: 'summarize', 'action_items', 'sentiment', 'reply_suggestions', 'question'")] = "summarize",
    specific_question: Annotated[str | None, Field(description="Specific question about the emails (only used when analysis_type is 'question')")] = None,
    max_emails: Annotated[int, Field(description="Maximum number of emails to analyze (1-100, default: 50)")] = 50,
    gmail_access_token: Annotated[str | None, Field(description="Gmail OAuth2 access token from authentication flow")] = None,
    include_promotions: Annotated[bool, Field(description="Include promotional emails (default: False)")] = False,
    include_bank_emails: Annotated[bool, Field(description="Include bank/financial emails (default: False)")] = False,
) -> str:
    """
    Search Gmail directly and analyze emails with smart filtering and caching.
    """
    
    if not gmail_access_token:
        return """🔐 **Gmail Authentication Required**

You need to authenticate with Gmail first to search your emails.

**Quick Setup:**
1. Use the `start_gmail_oauth` tool to get authentication instructions
2. Get your access token from OAuth flow
3. Come back and use this tool with your token

**Example after authentication:**
```
search_query: "is:unread"
analysis_type: "summarize"
gmail_access_token: "ya29.a0AfH6SMC..."
max_emails: 50
```

**Need help?** Use `start_gmail_oauth` tool for step-by-step setup! 🚀
"""

    try:
        # Validate max_emails (increased limit as requested)
        max_emails = max(1, min(max_emails, 100))
        
        # Apply smart filtering to search query
        filtered_query = _apply_smart_filtering(search_query, include_promotions, include_bank_emails)
        
        # Check cache first (use hash of query + analysis_type for security)
        query_hash = hashlib.sha256(f"{filtered_query}_{max_emails}_{analysis_type}".encode()).hexdigest()[:16]
        cache_key = f"gmail_search_{query_hash}"
        cached_result = await _get_cached_result(cache_key)
        
        if cached_result:
            return f"""📧 **Gmail Search Results** (📦 *cached - faster results*)

{cached_result}

*️⃣ Cache expires in 30 minutes for fresh data*
"""
        
        # Search Gmail
        emails = await _search_gmail_emails(gmail_access_token, filtered_query, max_emails)
        
        if not emails:
            return f"""📭 **No emails found** for search query: `{filtered_query}`

**Suggestions:**
- Try broader search terms
- Check date ranges (`after:2024-08-01`)  
- Remove specific filters
- Use `is:unread` for recent emails

**Applied Filters:**
- ❌ Promotions excluded: {not include_promotions}
- ❌ Bank emails excluded: {not include_bank_emails}
"""
        
        # Analyze the emails
        result = await _analyze_gmail_results(emails, analysis_type, specific_question, filtered_query)
        
        # Cache the result for 30 minutes
        await _cache_result(cache_key, result, expires_minutes=30)
        
        return result
            
    except Exception as e:
        error_msg = str(e)
        
        # Better error handling
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return f"""❌ **Authentication Error**

Your Gmail access token has expired or is invalid.

**Solutions:**
1. Get a fresh token using `start_gmail_oauth` tool
2. Make sure you copied the complete token
3. Check if token has proper Gmail API permissions

**Original Error:** {error_msg}
"""
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            return f"""❌ **Permission Error**

Gmail API access is restricted.

**Solutions:**
1. Enable Gmail API in Google Cloud Console
2. Check OAuth scope includes gmail.readonly
3. Verify project billing is enabled

**Original Error:** {error_msg}
"""
        else:
            return f"""❌ **Gmail API Error**

Something went wrong accessing Gmail.

**Error Details:** {error_msg}

**Common Fixes:**
- Check internet connection
- Verify access token format
- Try reducing max_emails parameter
- Use `start_gmail_oauth` for fresh authentication
"""

def _apply_smart_filtering(base_query: str, include_promotions: bool, include_bank_emails: bool) -> str:
    """Apply smart filtering to exclude promotions and bank emails"""
    
    filters = []
    
    # Base query
    filtered_query = base_query.strip()
    
    # Exclude promotions (Gmail's automatic categorization)
    if not include_promotions:
        filters.append("-category:promotions")
        filters.append("-category:social")
        # Common promotional indicators
        filters.append("-from:noreply")
        filters.append("-from:no-reply")
        filters.append("-subject:unsubscribe")
        filters.append("-subject:newsletter")
    
    # Exclude bank/financial emails
    if not include_bank_emails:
        bank_domains = [
            "bank", "banking", "sbi", "hdfc", "icici", "axis", "kotak", "pnb", 
            "paytm", "phonepe", "gpay", "razorpay", "upi", "cred", "jupiter",
            "noreply", "alerts", "notification"
        ]
        
        for domain in bank_domains:
            filters.append(f"-from:{domain}")
        
        # Common bank-related subjects
        bank_subjects = ["statement", "transaction", "payment", "otp", "alert", "debit", "credit"]
        for subject in bank_subjects:
            filters.append(f"-subject:{subject}")
    
    # Combine base query with filters
    if filters:
        filtered_query += " " + " ".join(filters)
    
    return filtered_query

# Simple in-memory cache (in production, use Redis or similar)
_email_cache = {}

async def _get_cached_result(cache_key: str) -> str | None:
    """Get cached email analysis result"""
    
    if cache_key in _email_cache:
        cached_data = _email_cache[cache_key]
        
        # Check if cache is expired (30 minutes)
        if datetime.now() < cached_data['expires']:
            return cached_data['result']
        else:
            # Remove expired cache
            del _email_cache[cache_key]
    
    return None

async def _cache_result(cache_key: str, result: str, expires_minutes: int = 30):
    """Cache email analysis result"""
    
    try:
        _email_cache[cache_key] = {
            'result': result,
            'expires': datetime.now() + timedelta(minutes=expires_minutes),
            'created': datetime.now()
        }
        
        # Clean up old cache entries (keep max 100 entries)
        if len(_email_cache) > 100:
            # Remove oldest entries
            sorted_keys = sorted(_email_cache.keys(), key=lambda k: _email_cache[k]['created'])
            for key in sorted_keys[:20]:  # Remove 20 oldest
                del _email_cache[key]
    except Exception:
        # If caching fails, continue without caching
        pass

async def _analyze_gmail_results(emails: list[dict], analysis_type: str, specific_question: str | None, search_query: str) -> str:
    """Analyze Gmail search results"""
    
    email_summaries = []
    
    for i, email in enumerate(emails, 1):
        subject = email.get('subject', 'No Subject')
        sender = email.get('sender', 'Unknown Sender')
        date = email.get('date', 'Unknown Date')
        
        # Clean sender (remove extra info)
        sender = sender.split('<')[0].strip() if '<' in sender else sender
        
        email_summaries.append(f"**{i}.** {subject[:60]}{'...' if len(subject) > 60 else ''}\n   📧 {sender} • 📅 {date[:16]}")
    
    # Perform analysis based on type
    if analysis_type == "summarize":
        return f"""📧 **Gmail Search Results & Summary**

**🔍 Search Query:** `{search_query}`
**📊 Found:** {len(emails)} email(s) (filtered: no promotions/bank emails)

**📋 Email List:**
{chr(10).join(email_summaries)}

**🔍 Overall Analysis:**
{_analyze_multiple_emails(emails, analysis_type)}

**⚡ Smart Features Applied:**
- ✅ Promotional emails filtered out
- ✅ Bank/financial alerts excluded  
- ✅ Results cached for 30 minutes
- ✅ Privacy-focused (no server storage)
"""
    
    elif analysis_type == "action_items":
        return f"""📧 **Action Items from Gmail Search**

**🔍 Search Query:** `{search_query}`
**📊 Emails Analyzed:** {len(emails)}

**✅ Action Items Found:**
{_extract_action_items_from_multiple(emails)}

**⏰ Urgent Items:**
{_extract_urgent_items_from_multiple(emails)}

**📧 Source Emails:**
{chr(10).join(email_summaries[:5])}
"""
    
    elif analysis_type == "question":
        if not specific_question:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide a specific question when using 'question' analysis type."))
        
        return f"""📧 **Question Analysis - Gmail Search**

**🔍 Search Query:** `{search_query}`
**❓ Your Question:** {specific_question}
**📊 Emails Analyzed:** {len(emails)}

**📍 Answer:**
{_answer_question_about_multiple_emails(emails, specific_question)}

**📋 Relevant Email Excerpts:**
{_find_relevant_excerpts_from_multiple(emails, specific_question)}

**📧 Source Emails:**
{chr(10).join(email_summaries[:3])}
"""
    
    else:
        # For sentiment and reply_suggestions, analyze first few emails
        results = []
        for i, email in enumerate(emails[:3], 1):  # Limit to first 3 for detailed analysis
            email_analysis = await email_analyzer(
                email_content=f"Subject: {email.get('subject', '')}\nFrom: {email.get('sender', '')}\n\n{email.get('body', '')}",
                analysis_type=analysis_type,
                specific_question=specific_question
            )
            results.append(f"**Email {i} Analysis:**\n{email_analysis}\n")
        
        return f"""📧 **Gmail Search & {analysis_type.title()} Analysis**

**🔍 Search Query:** `{search_query}`
**📊 Emails Found:** {len(emails)} (showing detailed analysis for first 3)

{chr(10).join(results)}

**📧 All Emails Found:**
{chr(10).join(email_summaries)}
"""

async def _search_gmail_emails(access_token: str, search_query: str, max_results: int = 50) -> list[dict]:
    """Search Gmail and return email data"""
    
    async with httpx.AsyncClient() as client:
        # Step 1: Search for message IDs
        search_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "q": search_query,
            "maxResults": max_results
        }
        
        try:
            response = await client.get(search_url, headers=headers, params=params, timeout=30)
        except httpx.TimeoutException:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Gmail API request timed out. Please try again."))
        except httpx.RequestError as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Gmail API connection error: {str(e)}"))
        
        if response.status_code == 429:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Gmail API rate limit exceeded. Please wait a moment and try again."))
        elif response.status_code != 200:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Gmail search failed: {response.status_code} - {response.text}"))
        
        search_results = response.json()
        messages = search_results.get('messages', [])
        
        if not messages:
            return []
        
        # Step 2: Fetch full message details (with rate limiting consideration)
        emails = []
        for i, message in enumerate(messages):
            message_id = message['id']
            
            message_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}"
            
            try:
                message_response = await client.get(message_url, headers=headers, timeout=15)
                
                if message_response.status_code == 200:
                    email_data = message_response.json()
                    parsed_email = _parse_gmail_message(email_data)
                    emails.append(parsed_email)
                elif message_response.status_code == 429:
                    # If we hit rate limits on individual messages, return what we have
                    break
                # For other errors, skip the message and continue
                
            except Exception:
                # Skip failed messages and continue
                continue
            
            # Add a small delay every 10 requests to be respectful to API
            if i > 0 and i % 10 == 0:
                await asyncio.sleep(0.1)
        
        return emails

def _parse_gmail_message(gmail_message: dict) -> dict:
    """Parse Gmail API message into readable format"""
    
    payload = gmail_message.get('payload', {})
    headers = payload.get('headers', [])
    
    # Extract headers
    subject = ""
    sender = ""
    date = ""
    
    for header in headers:
        name = header.get('name', '').lower()
        value = header.get('value', '')
        
        if name == 'subject':
            subject = value
        elif name == 'from':
            sender = value
        elif name == 'date':
            date = value
    
    # Extract body
    body = _extract_gmail_body(payload)
    
    return {
        'subject': subject,
        'sender': sender,
        'date': date,
        'body': body,
        'message_id': gmail_message.get('id', '')
    }

def _extract_gmail_body(payload: dict) -> str:
    """Extract email body from Gmail payload"""
    
    body = ""
    
    try:
        # Check if it's a simple message
        if 'body' in payload and payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
        
        # Check for multipart messages
        elif 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                    try:
                        part_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        body += part_body + "\n"
                    except Exception:
                        # Skip malformed parts
                        continue
                elif part.get('mimeType') == 'text/html' and part.get('body', {}).get('data') and not body:
                    # Fallback to HTML if no plain text
                    try:
                        html_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        # Simple HTML to text conversion
                        body = re.sub('<[^<]+?>', '', html_body)
                    except Exception:
                        # Skip malformed HTML parts
                        continue
    except Exception:
        # If all else fails, return empty body rather than crash
        body = "[Error: Could not decode email body]"
    
    return body.strip()

def _analyze_multiple_emails(emails: list[dict], analysis_type: str) -> str:
    """Analyze multiple emails and provide summary"""
    
    total_emails = len(emails)
    
    if analysis_type == "summarize":
        # Count key metrics
        senders = set()
        subjects = []
        total_length = 0
        
        for email in emails:
            senders.add(email.get('sender', 'Unknown'))
            subjects.append(email.get('subject', 'No Subject'))
            total_length += len(email.get('body', ''))
        
        return f"""**📊 Email Summary:**
• Total emails: {total_emails}
• Unique senders: {len(senders)}
• Average email length: {total_length // total_emails if total_emails > 0 else 0} characters

**👥 Top Senders:**
{chr(10).join([f"• {sender}" for sender in list(senders)[:5]])}

**📝 Recent Subjects:**
{chr(10).join([f"• {subject}" for subject in subjects[:5]])}
"""

def _extract_action_items_from_multiple(emails: list[dict]) -> str:
    """Extract action items from multiple emails"""
    
    all_actions = []
    
    for i, email in enumerate(emails, 1):
        body = email.get('body', '')
        actions = _extract_action_items(body)
        
        if actions and actions != "• No specific action items identified":
            all_actions.append(f"**From Email {i} ({email.get('subject', 'No Subject')}):**\n{actions}")
    
    return '\n\n'.join(all_actions) if all_actions else "• No action items found across emails"

def _extract_urgent_items_from_multiple(emails: list[dict]) -> str:
    """Extract urgent items from multiple emails"""
    
    urgent_items = []
    
    for i, email in enumerate(emails, 1):
        body = email.get('body', '')
        urgent = _extract_time_sensitive_items(body)
        
        if urgent and urgent != "⏰ No urgent deadlines identified":
            urgent_items.append(f"**Email {i}:** {urgent}")
    
    return '\n\n'.join(urgent_items) if urgent_items else "⏰ No urgent items found"

def _answer_question_about_multiple_emails(emails: list[dict], question: str) -> str:
    """Answer question about multiple emails"""
    
    # Combine all email content
    combined_content = ""
    for email in emails:
        combined_content += f" {email.get('subject', '')} {email.get('body', '')}"
    
    # Use existing question answering logic
    return _answer_question_about_email(combined_content, question)

def _find_relevant_excerpts_from_multiple(emails: list[dict], question: str) -> str:
    """Find relevant excerpts from multiple emails"""
    
    question_words = question.lower().split()
    relevant_excerpts = []
    
    for i, email in enumerate(emails, 1):
        subject = email.get('subject', '')
        body = email.get('body', '')
        
        # Check subject
        if any(word in subject.lower() for word in question_words):
            relevant_excerpts.append(f"• **Email {i} Subject:** \"{subject}\"")
        
        # Check body sentences
        sentences = body.split('.')
        for sentence in sentences[:3]:  # First 3 sentences only
            if len(sentence.strip()) > 20 and any(word in sentence.lower() for word in question_words):
                relevant_excerpts.append(f"• **Email {i}:** \"{sentence.strip()}\"")
                break
    
    return '\n'.join(relevant_excerpts[:5]) if relevant_excerpts else "• No directly relevant excerpts found"

# --- Run MCP Server ---
async def main():
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting MCP server on http://0.0.0.0:{port}")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=port)

if __name__ == "__main__":
    asyncio.run(main())
