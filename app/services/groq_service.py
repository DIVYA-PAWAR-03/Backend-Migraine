"""
Groq LLM Service for personalized migraine prevention suggestions.

This service integrates with Groq's API to generate AI-powered
health advice based on detected triggers and user health data.
"""

import logging
from typing import List, Optional
from groq import Groq
from datetime import datetime

from ..config import settings
from ..models.schemas import AISuggestionRequest, AISuggestionResponse, RiskLevel

logger = logging.getLogger(__name__)


class GroqService:
    """Service class for Groq LLM integration."""
    
    # System prompt for consistent AI behavior
    SYSTEM_PROMPT = """You are a specialized migraine prevention assistant with expertise in 
identifying triggers and providing actionable health advice. Your role is to:

1. Analyze the user's health data and detected triggers
2. Provide practical, immediate actions they can take
3. Be empathetic but concise
4. Focus on evidence-based recommendations
5. Never provide medical diagnoses or replace professional medical advice

Always remind users to consult healthcare professionals for persistent issues."""

    # Main prompt template for generating suggestions
    SUGGESTION_PROMPT_TEMPLATE = """Based on the following user health data and detected triggers, 
provide personalized migraine prevention advice:

═══════════════════════════════════════
📊 CURRENT HEALTH METRICS
═══════════════════════════════════════
• Risk Level: {risk_level}
• Stress Level: {stress_level}/10
• Sleep Hours: {sleep_hours} hours
• Heart Rate: {heart_rate} bpm
• Activity Level: {activity_level}/10
• Barometric Pressure: {weather_pressure} hPa
• Air Quality Index: {aqi}

═══════════════════════════════════════
⚠️ DETECTED TRIGGERS
═══════════════════════════════════════
{triggers_list}

═══════════════════════════════════════
📝 YOUR TASK
═══════════════════════════════════════
Provide 4-5 specific, actionable migraine prevention tips tailored to these triggers.
Format your response as:

SUMMARY: (One sentence overview of the situation)

RECOMMENDATIONS:
1. [Immediate action] - Brief explanation
2. [Immediate action] - Brief explanation
3. [Lifestyle adjustment] - Brief explanation
4. [Preventive measure] - Brief explanation
5. [When to seek help] - Brief guidance

Keep each recommendation under 30 words. Be practical and specific.
Mention urgency level: {urgency} priority based on risk level."""

    def __init__(self):
        """Initialize Groq client."""
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Groq client with API key."""
        try:
            if settings.GROQ_API_KEY:
                self.client = Groq(api_key=settings.GROQ_API_KEY)
                logger.info("Groq client initialized successfully")
            else:
                logger.warning("GROQ_API_KEY not set. AI suggestions will use fallback mode.")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
    
    def is_available(self) -> bool:
        """Check if Groq service is available."""
        return self.client is not None
    
    def _format_triggers(self, triggers: List[str]) -> str:
        """Format triggers list for the prompt."""
        if not triggers:
            return "• No specific triggers detected"
        return "\n".join([f"• {trigger}" for trigger in triggers])
    
    def _get_urgency(self, risk_level: RiskLevel) -> str:
        """Determine urgency based on risk level."""
        urgency_map = {
            RiskLevel.HIGH: "HIGH",
            RiskLevel.MEDIUM: "MODERATE", 
            RiskLevel.LOW: "LOW"
        }
        return urgency_map.get(risk_level, "MODERATE")
    
    def _parse_response(self, response_text: str) -> tuple[List[str], str]:
        """
        Parse the LLM response into structured suggestions.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Tuple of (suggestions list, summary)
        """
        suggestions = []
        summary = ""
        
        lines = response_text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            
            # Extract summary
            if line.upper().startswith("SUMMARY:"):
                summary = line[8:].strip()
            
            # Extract numbered recommendations
            elif line and line[0].isdigit() and "." in line[:3]:
                # Remove the number and period
                suggestion = line.split(".", 1)[1].strip() if "." in line else line
                if suggestion:
                    suggestions.append(suggestion)
        
        # Fallback if parsing failed
        if not suggestions:
            suggestions = [response_text[:200] + "..."] if len(response_text) > 200 else [response_text]
        
        if not summary:
            summary = "Based on your health data, here are personalized recommendations."
        
        return suggestions, summary
    
    def _get_fallback_suggestions(self, request: AISuggestionRequest) -> AISuggestionResponse:
        """
        Generate fallback suggestions when Groq is unavailable.
        
        Args:
            request: AI suggestion request data
            
        Returns:
            AISuggestionResponse with generic suggestions
        """
        suggestions = []
        
        # Generate suggestions based on triggers
        if request.stress_level >= 7:
            suggestions.append("Practice deep breathing exercises for 5-10 minutes to reduce stress levels.")
        
        if request.sleep_hours < 6:
            suggestions.append("Aim for 7-8 hours of sleep tonight. Consider a bedtime routine without screens.")
        
        if request.heart_rate > 90:
            suggestions.append("Take a break and practice relaxation. Avoid caffeine and stimulants.")
        
        if request.activity_level < 4:
            suggestions.append("Try a gentle 15-minute walk to improve circulation and reduce tension.")
        
        if request.aqi > 100:
            suggestions.append("Stay indoors if possible. Use air purification and stay hydrated.")
        
        if request.weather_pressure < 1000 or request.weather_pressure > 1025:
            suggestions.append("Weather changes detected. Stay hydrated and maintain regular meal times.")
        
        # Add general advice if few triggers
        if len(suggestions) < 3:
            suggestions.extend([
                "Stay well-hydrated by drinking at least 8 glasses of water today.",
                "Maintain regular meal times to keep blood sugar stable.",
                "If symptoms worsen, consult a healthcare professional."
            ])
        
        urgency = self._get_urgency(request.risk_level)
        
        return AISuggestionResponse(
            suggestions=suggestions[:5],
            summary=f"Based on your {request.risk_level.value} risk level and detected triggers, here are personalized recommendations.",
            urgency=urgency.lower()
        )
    
    async def get_suggestions(self, request: AISuggestionRequest) -> AISuggestionResponse:
        """
        Generate AI-powered migraine prevention suggestions.
        
        Args:
            request: AISuggestionRequest with user health data and triggers
            
        Returns:
            AISuggestionResponse with personalized suggestions
        """
        # Use fallback if Groq is not available
        if not self.is_available():
            logger.info("Using fallback suggestions (Groq unavailable)")
            return self._get_fallback_suggestions(request)
        
        try:
            # Prepare the prompt
            urgency = self._get_urgency(request.risk_level)
            triggers_formatted = self._format_triggers(request.triggers)
            
            prompt = self.SUGGESTION_PROMPT_TEMPLATE.format(
                risk_level=request.risk_level.value,
                stress_level=request.stress_level,
                sleep_hours=request.sleep_hours,
                heart_rate=request.heart_rate,
                activity_level=request.activity_level,
                weather_pressure=request.weather_pressure,
                aqi=request.aqi,
                triggers_list=triggers_formatted,
                urgency=urgency
            )
            
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model=settings.GROQ_MODEL,
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            
            response_text = chat_completion.choices[0].message.content
            logger.info("Successfully generated AI suggestions")
            
            # Parse the response
            suggestions, summary = self._parse_response(response_text)
            
            return AISuggestionResponse(
                suggestions=suggestions[:5],
                summary=summary,
                urgency=urgency.lower()
            )
            
        except Exception as e:
            logger.error(f"Error generating AI suggestions: {e}")
            return self._get_fallback_suggestions(request)
    
    def get_prompt_template(self) -> str:
        """Return the prompt template for documentation purposes."""
        return self.SUGGESTION_PROMPT_TEMPLATE
    
    async def chat(self, request) -> 'ChatResponse':
        """
        Interactive chat about migraine topics.
        
        Args:
            request: ChatRequest with user message
            
        Returns:
            ChatResponse with AI reply
        """
        from ..models.schemas import ChatResponse
        
        # Medical chatbot system prompt
        chat_system_prompt = """You are a knowledgeable and empathetic migraine health assistant. Your role is to:

1. Answer questions about migraines, triggers, symptoms, and prevention
2. Provide evidence-based health information
3. Offer practical lifestyle and wellness advice
4. Be supportive and understanding of migraine sufferers
5. Always recommend consulting healthcare professionals for medical decisions

Important guidelines:
- Never diagnose conditions or prescribe medications
- Be clear about the difference between general information and medical advice
- If someone describes emergency symptoms (sudden severe headache, confusion, vision loss), advise immediate medical attention
- Keep responses helpful, concise, and actionable
- Use friendly, accessible language

Topics you can help with:
- Understanding different types of migraines
- Common triggers (stress, sleep, diet, weather, hormones)
- Prevention strategies and lifestyle modifications
- When to seek medical care
- Managing migraine episodes
- Tracking and identifying personal triggers"""

        if not self.is_available():
            return self._get_fallback_chat_response(request.message)
        
        try:
            # Build context-aware message
            user_message = request.message
            if request.context:
                context_info = f"\n[Context: User's recent prediction was {request.context.get('recent_prediction', 'unknown')}]"
                user_message = context_info + "\n\nUser question: " + user_message
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": chat_system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=settings.GROQ_MODEL,
                temperature=0.7,
                max_tokens=600,
                top_p=0.9
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # Generate follow-up suggestions
            suggestions = self._generate_follow_up_suggestions(request.message, response_text)
            related_topics = self._get_related_topics(request.message)
            
            return ChatResponse(
                response=response_text,
                suggestions=suggestions,
                related_topics=related_topics
            )
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return self._get_fallback_chat_response(request.message)
    
    def _get_fallback_chat_response(self, message: str) -> 'ChatResponse':
        """Generate fallback response when Groq unavailable."""
        from ..models.schemas import ChatResponse
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['trigger', 'cause', 'why']):
            response = """Common migraine triggers include:

1. **Stress** - One of the most common triggers
2. **Sleep changes** - Too little or too much sleep
3. **Hormonal changes** - Especially in women
4. **Weather changes** - Barometric pressure shifts
5. **Diet factors** - Skipping meals, dehydration, certain foods
6. **Sensory stimuli** - Bright lights, strong smells

Keeping a migraine diary can help identify your personal triggers. Would you like tips on tracking triggers?"""
        
        elif any(word in message_lower for word in ['symptom', 'feel', 'sign']):
            response = """Common migraine symptoms include:

**Before the headache (Prodrome):**
- Mood changes, food cravings, neck stiffness

**Aura (if present):**
- Visual disturbances, tingling sensations

**During the headache:**
- Throbbing pain (often one-sided)
- Nausea and vomiting
- Sensitivity to light and sound

**After (Postdrome):**
- Fatigue, confusion, or euphoria

If you experience sudden severe headaches or new symptoms, seek immediate medical attention."""
        
        elif any(word in message_lower for word in ['prevent', 'stop', 'avoid']):
            response = """Migraine prevention strategies:

1. **Maintain regular sleep schedule** - Same bedtime and wake time
2. **Stay hydrated** - At least 8 glasses of water daily
3. **Manage stress** - Try meditation, yoga, or deep breathing
4. **Regular exercise** - 30 minutes of moderate activity
5. **Avoid known triggers** - Keep a diary to identify them
6. **Don't skip meals** - Maintain stable blood sugar

For frequent migraines (4+ per month), consult a doctor about preventive medications."""
        
        else:
            response = """I'm here to help with migraine-related questions! I can assist with:

• Understanding migraine types and symptoms
• Identifying and avoiding triggers
• Prevention strategies and lifestyle tips
• When to seek medical care
• Managing migraine episodes

What would you like to know more about?"""
        
        return ChatResponse(
            response=response,
            suggestions=[
                "What are common migraine triggers?",
                "How can I prevent migraines?",
                "When should I see a doctor?"
            ],
            related_topics=["Triggers", "Prevention", "Symptoms", "Treatment"]
        )
    
    def _generate_follow_up_suggestions(self, user_message: str, response: str) -> List[str]:
        """Generate relevant follow-up questions."""
        message_lower = user_message.lower()
        
        if 'trigger' in message_lower:
            return [
                "How do I track my triggers?",
                "What foods trigger migraines?",
                "How does weather affect migraines?"
            ]
        elif 'symptom' in message_lower:
            return [
                "What is a migraine aura?",
                "When is a headache an emergency?",
                "How are migraines diagnosed?"
            ]
        elif 'prevent' in message_lower:
            return [
                "What lifestyle changes help most?",
                "Are there preventive medications?",
                "How does exercise help migraines?"
            ]
        else:
            return [
                "What causes migraines?",
                "How can I prevent migraines?",
                "What should I do during an attack?"
            ]
    
    def _get_related_topics(self, message: str) -> List[str]:
        """Get related topics based on the question."""
        message_lower = message.lower()
        
        all_topics = ["Triggers", "Prevention", "Symptoms", "Treatment", "Lifestyle", "Diet", "Sleep", "Stress", "Weather", "Medication"]
        
        # Return topics not directly mentioned
        related = []
        for topic in all_topics:
            if topic.lower() not in message_lower:
                related.append(topic)
            if len(related) >= 4:
                break
        
        return related


# Singleton instance
groq_service = GroqService()
