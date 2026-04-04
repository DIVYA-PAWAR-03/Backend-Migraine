"""
PDF Report Generation Service for Migraine Tracker.

Generates daily and weekly migraine reports in PDF format.
"""

import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import logging

logger = logging.getLogger(__name__)


class ReportService:
    """Service for generating PDF reports."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            alignment=TA_CENTER,
            spaceAfter=20
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.gray,
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceBefore=20,
            spaceAfter=10
        ))
        
        # Risk High style
        self.styles.add(ParagraphStyle(
            name='RiskHigh',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#ef4444'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Risk Medium style
        self.styles.add(ParagraphStyle(
            name='RiskMedium',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#f59e0b'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Risk Low style
        self.styles.add(ParagraphStyle(
            name='RiskLow',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#10b981'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
    
    def _get_risk_style(self, risk_level: str) -> str:
        """Get the appropriate style name for a risk level."""
        risk_map = {
            'high': 'RiskHigh',
            'medium': 'RiskMedium',
            'low': 'RiskLow'
        }
        return risk_map.get(risk_level.lower(), 'Normal')
    
    def _get_risk_color(self, risk_level: str) -> colors.Color:
        """Get color for risk level."""
        color_map = {
            'high': colors.HexColor('#ef4444'),
            'medium': colors.HexColor('#f59e0b'),
            'low': colors.HexColor('#10b981')
        }
        return color_map.get(risk_level.lower(), colors.gray)
    
    def generate_daily_report(
        self,
        prediction_data: Dict,
        health_data: Dict,
        ai_suggestions: List[str] = None,
        user_name: str = "User",
        patient_info: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Generate a daily migraine risk report.
        
        Args:
            prediction_data: Prediction results (risk_level, probability, triggers)
            health_data: Input health metrics
            ai_suggestions: AI-generated suggestions
            user_name: Name for the report
            
        Returns:
            PDF file as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        story = []
        
        # Header
        story.append(Paragraph("🧠 Migraine Risk Report", self.styles['ReportTitle']))
        story.append(Paragraph(
            f"Daily Assessment - {datetime.now().strftime('%B %d, %Y')}",
            self.styles['ReportSubtitle']
        ))
        story.append(Paragraph(f"Patient: {user_name}", self.styles['Normal']))
        story.append(Paragraph(
            f"Patient ID: {(patient_info or {}).get('patient_id', 'N/A')}",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"Email: {(patient_info or {}).get('email', 'N/A')}",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"Age/Gender: {(patient_info or {}).get('age', 'N/A')} / {(patient_info or {}).get('gender', 'N/A')}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 20))
        
        # Divider
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#667eea')))
        story.append(Spacer(1, 20))
        
        # Risk Assessment Section
        story.append(Paragraph("📊 Risk Assessment", self.styles['SectionHeader']))
        
        risk_level = prediction_data.get('risk_level', 'Unknown')
        probability = prediction_data.get('probability', 0)
        
        # Risk level display
        risk_style = self._get_risk_style(risk_level)
        story.append(Paragraph(f"Risk Level: {risk_level.upper()}", self.styles[risk_style]))
        story.append(Paragraph(
            f"Probability: {probability * 100:.1f}%",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 20))
        
        # Health Metrics Table
        story.append(Paragraph("📋 Health Metrics Analyzed", self.styles['SectionHeader']))
        
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Sleep Hours', f"{health_data.get('sleep_hours', 'N/A')} hrs", 
             self._get_metric_status('sleep', health_data.get('sleep_hours', 7))],
            ['Stress Level', f"{health_data.get('stress_level', 'N/A')}/10",
             self._get_metric_status('stress', health_data.get('stress_level', 5))],
            ['Heart Rate', f"{health_data.get('heart_rate', 'N/A')} bpm",
             self._get_metric_status('heart_rate', health_data.get('heart_rate', 70))],
            ['Activity Level', f"{health_data.get('activity_level', 'N/A')}/10",
             self._get_metric_status('activity', health_data.get('activity_level', 5))],
            ['Weather Pressure', f"{health_data.get('weather_pressure', 'N/A')} hPa",
             self._get_metric_status('pressure', health_data.get('weather_pressure', 1013))],
            ['Air Quality Index', f"{health_data.get('aqi', 'N/A')}",
             self._get_metric_status('aqi', health_data.get('aqi', 50))],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Triggers Section
        triggers = prediction_data.get('triggers', [])
        if triggers:
            story.append(Paragraph("⚠️ Detected Triggers", self.styles['SectionHeader']))
            trigger_items = [ListItem(Paragraph(t, self.styles['Normal'])) for t in triggers]
            story.append(ListFlowable(trigger_items, bulletType='bullet'))
            story.append(Spacer(1, 20))
        
        # AI Suggestions Section
        if ai_suggestions:
            story.append(Paragraph("💡 AI Recommendations", self.styles['SectionHeader']))
            suggestion_items = [ListItem(Paragraph(s, self.styles['Normal'])) for s in ai_suggestions]
            story.append(ListFlowable(suggestion_items, bulletType='bullet'))
            story.append(Spacer(1, 20))
        
        # Prevention Tips
        story.append(Paragraph("🛡️ Prevention Tips", self.styles['SectionHeader']))
        prevention_tips = self._get_prevention_tips(risk_level, triggers)
        tip_items = [ListItem(Paragraph(tip, self.styles['Normal'])) for tip in prevention_tips]
        story.append(ListFlowable(tip_items, bulletType='bullet'))
        story.append(Spacer(1, 30))
        
        # Footer
        story.append(HRFlowable(width="100%", thickness=1, color=colors.gray))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Migraine AI",
            ParagraphStyle(name='Footer', fontSize=8, textColor=colors.gray, alignment=TA_CENTER)
        ))
        story.append(Paragraph(
            "⚠️ This report is for informational purposes only and should not replace professional medical advice.",
            ParagraphStyle(name='Disclaimer', fontSize=8, textColor=colors.gray, alignment=TA_CENTER)
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def generate_weekly_report(
        self,
        weekly_data: List[Dict],
        summary_stats: Dict = None,
        user_name: str = "User",
        patient_info: Optional[Dict[str, Any]] = None,
        previous_history_summary: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Generate a weekly migraine analysis report.
        
        Args:
            weekly_data: List of daily prediction/health data for the week
            summary_stats: Aggregated statistics for the week
            user_name: Name for the report
            
        Returns:
            PDF file as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        story = []
        
        # Header
        story.append(Paragraph("🧠 Weekly Migraine Report", self.styles['ReportTitle']))
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=6)
        story.append(Paragraph(
            f"{start_date.strftime('%B %d')} - {end_date.strftime('%B %d, %Y')}",
            self.styles['ReportSubtitle']
        ))
        story.append(Paragraph(f"Patient: {user_name}", self.styles['Normal']))
        story.append(Paragraph(
            f"Patient ID: {(patient_info or {}).get('patient_id', 'N/A')}",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"Email: {(patient_info or {}).get('email', 'N/A')}",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"Age/Gender: {(patient_info or {}).get('age', 'N/A')} / {(patient_info or {}).get('gender', 'N/A')}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 20))
        
        # Divider
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#667eea')))
        story.append(Spacer(1, 20))
        
        # Weekly Summary Section
        story.append(Paragraph("📈 Weekly Summary", self.styles['SectionHeader']))
        
        if summary_stats:
            summary_data = [
                ['Metric', 'Value'],
                ['Total Assessments', str(summary_stats.get('total_assessments', len(weekly_data)))],
                ['Average Risk Score', f"{summary_stats.get('avg_probability', 0) * 100:.1f}%"],
                ['High Risk Days', str(summary_stats.get('high_risk_days', 0))],
                ['Medium Risk Days', str(summary_stats.get('medium_risk_days', 0))],
                ['Low Risk Days', str(summary_stats.get('low_risk_days', 0))],
                ['Most Common Trigger', summary_stats.get('top_trigger', 'None detected')],
            ]
        else:
            # Calculate from weekly_data
            high_days = sum(1 for d in weekly_data if d.get('risk_level', '').lower() == 'high')
            medium_days = sum(1 for d in weekly_data if d.get('risk_level', '').lower() == 'medium')
            low_days = sum(1 for d in weekly_data if d.get('risk_level', '').lower() == 'low')
            avg_prob = sum(d.get('probability', 0) for d in weekly_data) / max(len(weekly_data), 1)
            
            # Count triggers
            all_triggers = []
            for d in weekly_data:
                all_triggers.extend(d.get('triggers', []))
            top_trigger = max(set(all_triggers), key=all_triggers.count) if all_triggers else 'None detected'
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Assessments', str(len(weekly_data))],
                ['Average Risk Score', f"{avg_prob * 100:.1f}%"],
                ['High Risk Days', str(high_days)],
                ['Medium Risk Days', str(medium_days)],
                ['Low Risk Days', str(low_days)],
                ['Most Common Trigger', top_trigger[:30] if len(top_trigger) > 30 else top_trigger],
            ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 30))

        if previous_history_summary:
            story.append(Paragraph("🧠 Historical Memory (Last 90 Days)", self.styles['SectionHeader']))
            memory_data = [
                ['Metric', 'Value'],
                ['Total Logs', str(previous_history_summary.get('total_records', 0))],
                ['Confirmed Migraine Days', str(previous_history_summary.get('migraine_count', 0))],
                ['Average Risk', f"{previous_history_summary.get('average_risk', 0) * 100:.1f}%"],
                ['Top Trigger', previous_history_summary.get('top_trigger', 'None detected')],
            ]

            memory_table = Table(memory_data, colWidths=[2.5*inch, 2.5*inch])
            memory_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecfeff')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#99f6e4')),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(memory_table)
            story.append(Spacer(1, 30))
        
        # Daily Breakdown
        story.append(Paragraph("📅 Daily Breakdown", self.styles['SectionHeader']))
        
        if weekly_data:
            daily_data = [['Day', 'Risk Level', 'Probability', 'Triggers']]
            for i, day in enumerate(weekly_data[-7:]):  # Last 7 days
                day_name = (datetime.now() - timedelta(days=len(weekly_data)-1-i)).strftime('%a %m/%d')
                risk = day.get('risk_level', 'N/A')
                prob = f"{day.get('probability', 0) * 100:.0f}%"
                triggers = ', '.join(day.get('triggers', [])[:2]) or 'None'
                if len(triggers) > 25:
                    triggers = triggers[:22] + '...'
                daily_data.append([day_name, risk.capitalize(), prob, triggers])
            
            daily_table = Table(daily_data, colWidths=[1*inch, 1.2*inch, 1*inch, 2.3*inch])
            daily_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(daily_table)
        else:
            story.append(Paragraph("No daily data available for this week.", self.styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Trigger Analysis
        story.append(Paragraph("🎯 Trigger Analysis", self.styles['SectionHeader']))
        
        all_triggers = []
        for d in weekly_data:
            all_triggers.extend(d.get('triggers', []))
        
        if all_triggers:
            trigger_counts = {}
            for t in all_triggers:
                trigger_counts[t] = trigger_counts.get(t, 0) + 1
            
            sorted_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            trigger_data = [['Trigger', 'Occurrences']]
            for trigger, count in sorted_triggers:
                trigger_text = trigger[:40] + '...' if len(trigger) > 40 else trigger
                trigger_data.append([trigger_text, str(count)])
            
            trigger_table = Table(trigger_data, colWidths=[4*inch, 1.5*inch])
            trigger_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f59e0b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fef3c7')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#fcd34d')),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(trigger_table)
        else:
            story.append(Paragraph("No triggers detected this week.", self.styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Weekly Recommendations
        story.append(Paragraph("💡 Weekly Recommendations", self.styles['SectionHeader']))
        recommendations = self._get_weekly_recommendations(weekly_data)
        rec_items = [ListItem(Paragraph(rec, self.styles['Normal'])) for rec in recommendations]
        story.append(ListFlowable(rec_items, bulletType='bullet'))
        story.append(Spacer(1, 30))
        
        # Footer
        story.append(HRFlowable(width="100%", thickness=1, color=colors.gray))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Migraine AI",
            ParagraphStyle(name='Footer', fontSize=8, textColor=colors.gray, alignment=TA_CENTER)
        ))
        story.append(Paragraph(
            "⚠️ This report is for informational purposes only and should not replace professional medical advice.",
            ParagraphStyle(name='Disclaimer', fontSize=8, textColor=colors.gray, alignment=TA_CENTER)
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _get_metric_status(self, metric: str, value: float) -> str:
        """Get status text for a metric value."""
        if metric == 'sleep':
            if value >= 7:
                return '✅ Good'
            elif value >= 5:
                return '⚠️ Moderate'
            else:
                return '❌ Poor'
        elif metric == 'stress':
            if value <= 3:
                return '✅ Low'
            elif value <= 6:
                return '⚠️ Moderate'
            else:
                return '❌ High'
        elif metric == 'heart_rate':
            if value <= 80:
                return '✅ Normal'
            elif value <= 95:
                return '⚠️ Elevated'
            else:
                return '❌ High'
        elif metric == 'activity':
            if value >= 6:
                return '✅ Active'
            elif value >= 3:
                return '⚠️ Moderate'
            else:
                return '❌ Low'
        elif metric == 'pressure':
            if 1005 <= value <= 1020:
                return '✅ Normal'
            else:
                return '⚠️ Unusual'
        elif metric == 'aqi':
            if value <= 50:
                return '✅ Good'
            elif value <= 100:
                return '⚠️ Moderate'
            else:
                return '❌ Poor'
        return '—'
    
    def _get_prevention_tips(self, risk_level: str, triggers: List[str]) -> List[str]:
        """Get prevention tips based on risk level and triggers."""
        tips = []
        
        if risk_level.lower() == 'high':
            tips.extend([
                "Consider taking preventive medication if prescribed",
                "Avoid known triggers today",
                "Rest in a quiet, dark environment if symptoms appear",
                "Stay very well hydrated",
                "Avoid strenuous activities"
            ])
        elif risk_level.lower() == 'medium':
            tips.extend([
                "Monitor for early warning signs",
                "Maintain regular meal and sleep schedule",
                "Practice relaxation techniques",
                "Keep rescue medication nearby"
            ])
        else:
            tips.extend([
                "Continue healthy habits",
                "Maintain regular sleep schedule",
                "Stay active with moderate exercise",
                "Keep tracking your triggers"
            ])
        
        # Add trigger-specific tips
        trigger_text = ' '.join(triggers).lower()
        if 'stress' in trigger_text:
            tips.append("Practice deep breathing or meditation today")
        if 'sleep' in trigger_text:
            tips.append("Prioritize getting 7-8 hours of sleep tonight")
        if 'air quality' in trigger_text or 'aqi' in trigger_text:
            tips.append("Consider using an air purifier or staying indoors")
        
        return tips[:6]  # Limit to 6 tips
    
    def _get_weekly_recommendations(self, weekly_data: List[Dict]) -> List[str]:
        """Generate weekly recommendations based on patterns."""
        recommendations = []
        
        if not weekly_data:
            return ["Start tracking your daily health metrics to get personalized recommendations"]
        
        # Analyze patterns
        high_days = sum(1 for d in weekly_data if d.get('risk_level', '').lower() == 'high')
        
        if high_days >= 3:
            recommendations.append("You had multiple high-risk days - consider consulting with your healthcare provider")
        
        # Check for common triggers
        all_triggers = []
        for d in weekly_data:
            all_triggers.extend(d.get('triggers', []))
        
        trigger_text = ' '.join(all_triggers).lower()
        
        if 'stress' in trigger_text:
            recommendations.append("Stress appears frequently - consider incorporating daily relaxation practices")
        if 'sleep' in trigger_text:
            recommendations.append("Sleep issues detected - try to establish a consistent bedtime routine")
        if 'activity' in trigger_text:
            recommendations.append("Activity levels vary - aim for regular moderate exercise throughout the week")
        
        recommendations.extend([
            "Keep maintaining your migraine diary for better pattern recognition",
            "Stay consistent with preventive measures that work for you",
            "Review this week's triggers to plan better for next week"
        ])
        
        return recommendations[:5]


# Singleton instance
report_service = ReportService()
