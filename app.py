"""
Kalapurusha Vedic Astrology - Streamlit Web Application
Upload kundali screenshot → Get instant analysis with Graha Drishti
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import re
import pandas as pd
from io import BytesIO
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import base64

# ==================== VEDIC ASTROLOGY CONSTANTS ====================

class Planet(Enum):
    SUN = "Sun"
    MOON = "Moon"
    MARS = "Mars"
    MERCURY = "Mercury"
    JUPITER = "Jupiter"
    VENUS = "Venus"
    SATURN = "Saturn"
    RAHU = "Rahu"
    KETU = "Ketu"

PLANET_COLORS = {
    Planet.SUN: "#FF0000",
    Planet.MOON: "#E0E0E0",
    Planet.MARS: "#FF4500",
    Planet.MERCURY: "#00FF00",
    Planet.JUPITER: "#FFD700",
    Planet.VENUS: "#FFB6C1",
    Planet.SATURN: "#0000FF",
    Planet.RAHU: "#696969",
    Planet.KETU: "#8B4513"
}

# Enhanced planet abbreviations for better OCR matching
PLANET_ABBREVIATIONS = {
    'SU': Planet.SUN, 'SUN': Planet.SUN,
    'MO': Planet.MOON, 'MOON': Planet.MOON,
    'MA': Planet.MARS, 'MARS': Planet.MARS, 'MAN': Planet.MARS,
    'ME': Planet.MERCURY, 'MERC': Planet.MERCURY,
    'JU': Planet.JUPITER, 'JUP': Planet.JUPITER,
    'VE': Planet.VENUS, 'VEN': Planet.VENUS,
    'SA': Planet.SATURN, 'SAT': Planet.SATURN,
    'RA': Planet.RAHU, 'RAH': Planet.RAHU,
    'KE': Planet.KETU, 'KET': Planet.KETU,
    'NE': Planet.MERCURY,  # Common OCR error
    'PI': Planet.JUPITER,  # Pl → Pi OCR error
    'UF': Planet.VENUS     # Uf → Ve OCR error  
}

GRAHA_DRISHTI = {
    Planet.SUN: [7],
    Planet.MOON: [7],
    Planet.MARS: [4, 7, 8],
    Planet.MERCURY: [7],
    Planet.JUPITER: [5, 7, 9],
    Planet.VENUS: [7],
    Planet.SATURN: [3, 7, 10],
    Planet.RAHU: [5, 7, 9],
    Planet.KETU: [5, 7, 9]
}

KALAPURUSHA_BODY_PARTS = {
    1: "Head (Krutthika)",
    2: "Face/Mouth (Aarudra)",
    3: "Throat/Shoulders (Punarwasu)",
    4: "Chest/Heart (Aaslesha)",
    5: "Stomach (Pubba)",
    6: "Intestines (Jyeshta)",
    7: "Lower Abdomen (Hastha)",
    8: "Genitals (Anuradha)",
    9: "Thighs (Poorvashaadha)",
    10: "Knees (Sravanam)",
    11: "Calves/Legs (Satabhisham)",
    12: "Feet (Uttarabhadra)"
}

HOUSE_SIGNIFICATIONS = {
    1: "Self, Physical Body, Vitality, Head",
    2: "Wealth, Speech, Family, Face",
    3: "Courage, Siblings, Communication, Throat",
    4: "Mother, Home, Heart, Chest",
    5: "Intelligence, Children, Stomach",
    6: "Disease, Enemies, Intestines",
    7: "Spouse, Partnerships, Lower Abdomen",
    8: "Longevity, Transformation, Genitals",
    9: "Fortune, Dharma, Thighs",
    10: "Career, Status, Knees",
    11: "Gains, Income, Calves",
    12: "Loss, Liberation, Feet"
}

PLANET_NATURE = {
    Planet.SUN: "Malefic (Soul significator)",
    Planet.MOON: "Benefic (if waxing)",
    Planet.MARS: "Malefic (Strength giver)",
    Planet.MERCURY: "Neutral",
    Planet.JUPITER: "Great Benefic",
    Planet.VENUS: "Benefic",
    Planet.SATURN: "Malefic (Karma)",
    Planet.RAHU: "Malefic (Shadow)",
    Planet.KETU: "Malefic (Moksha)"
}

PLANET_GEMSTONES = {
    Planet.SUN: "Ruby", Planet.MOON: "Pearl", Planet.MARS: "Red Coral",
    Planet.MERCURY: "Emerald", Planet.JUPITER: "Yellow Sapphire",
    Planet.VENUS: "Diamond", Planet.SATURN: "Blue Sapphire",
    Planet.RAHU: "Hessonite", Planet.KETU: "Cat's Eye"
}

PLANET_GRAINS = {
    Planet.SUN: "Wheat", Planet.MOON: "Rice",
    Planet.MARS: "Red Lentils", Planet.MERCURY: "Green Gram",
    Planet.JUPITER: "Chana Dal", Planet.VENUS: "White Rice",
    Planet.SATURN: "Black Sesame", Planet.RAHU: "Black Lentils",
    Planet.KETU: "Multi-grain"
}

@dataclass
class PlanetPosition:
    planet: Planet
    house: int
    degrees: Optional[float] = None
    
# ==================== ENHANCED OCR ENGINE ====================

class EnhancedKundaliOCR:
    """Enhanced OCR for various kundali formats"""
    
    def __init__(self):
        # North Indian diamond house positions (normalized 0-1)
        self.house_regions = {
            1: (0.38, 0.42, 0.62, 0.58),   # Center
            2: (0.32, 0.12, 0.48, 0.32),   # Top-left
            3: (0.08, 0.12, 0.28, 0.32),   # Left-top
            4: (0.03, 0.32, 0.23, 0.48),   # Left-center
            5: (0.08, 0.52, 0.28, 0.72),   # Left-bottom
            6: (0.32, 0.68, 0.48, 0.88),   # Bottom-left
            7: (0.38, 0.72, 0.62, 0.88),   # Bottom-center
            8: (0.52, 0.68, 0.68, 0.88),   # Bottom-right
            9: (0.72, 0.52, 0.92, 0.72),   # Right-bottom
            10: (0.77, 0.32, 0.97, 0.48),  # Right-center
            11: (0.72, 0.12, 0.92, 0.32),  # Right-top
            12: (0.52, 0.12, 0.68, 0.32)   # Top-right
        }
    
    def extract_from_image(self, image) -> List[PlanetPosition]:
        """Extract planets from uploaded image"""
        # Convert PIL to numpy
        img = np.array(image)
        
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        height, width = img.shape[:2]
        positions = []
        
        for house_num, (x1, y1, x2, y2) in self.house_regions.items():
            # Convert to pixels
            px1, py1 = int(x1 * width), int(y1 * height)
            px2, py2 = int(x2 * width), int(y2 * height)
            
            # Extract region
            roi = img[py1:py2, px1:px2]
            
            # Preprocess for better OCR
            roi = self._preprocess_roi(roi)
            
            # OCR with multiple configs
            text = self._robust_ocr(roi)
            
            # Parse planets
            planets = self._parse_planets(text, house_num)
            positions.extend(planets)
        
        return positions
    
    def _preprocess_roi(self, roi):
        """Enhance image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _robust_ocr(self, roi):
        """OCR with multiple attempts"""
        configs = [
            '--psm 6',
            '--psm 11',
            '--psm 12'
        ]
        
        best_text = ""
        for config in configs:
            try:
                text = pytesseract.image_to_string(roi, config=config)
                if len(text) > len(best_text):
                    best_text = text
            except:
                continue
        
        return best_text.upper()
    
    def _parse_planets(self, text: str, house: int) -> List[PlanetPosition]:
        """Parse planets with fuzzy matching"""
        positions = []
        text = text.replace('\n', ' ').replace('  ', ' ')
        
        # Try each abbreviation
        for abbr, planet in PLANET_ABBREVIATIONS.items():
            if abbr in text:
                # Extract degrees if present
                degree_pattern = rf'{abbr}\s*(\d+)'
                match = re.search(degree_pattern, text)
                
                degrees = None
                if match:
                    degrees = float(match.group(1))
                
                positions.append(PlanetPosition(
                    planet=planet,
                    house=house,
                    degrees=degrees
                ))
        
        return positions

# ==================== CHART GENERATOR ====================

class KalapurushaChartGenerator:
    """Generate beautiful Kalapurusha chart with aspects"""
    
    def __init__(self, size: int = 1000):
        self.size = size
        self.center = size // 2
        self.radius = size * 0.38
    
    def generate(self, positions: List[PlanetPosition]) -> Image.Image:
        """Generate complete chart"""
        img = Image.new('RGB', (self.size, self.size), 'white')
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Draw structure
        self._draw_diamond(draw)
        self._draw_house_labels(draw)
        
        # Place planets
        self._draw_planets(draw, positions)
        
        # Draw aspects
        self._draw_aspects(draw, positions)
        
        return img
    
    def _draw_diamond(self, draw: ImageDraw.Draw):
        """Draw diamond structure"""
        r = self.radius
        c = self.center
        
        # Four corners
        top = (c, c - r)
        right = (c + r, c)
        bottom = (c, c + r)
        left = (c - r, c)
        
        # Outer diamond
        draw.polygon([top, right, bottom, left], outline='black', width=4)
        
        # Cross lines
        draw.line([left, right], fill='black', width=2)
        draw.line([top, bottom], fill='black', width=2)
        
        # Diagonal divisions
        draw.line([top, left], fill='black', width=2)
        draw.line([top, right], fill='black', width=2)
        draw.line([bottom, left], fill='black', width=2)
        draw.line([bottom, right], fill='black', width=2)
    
    def _get_house_center(self, house: int) -> Tuple[int, int]:
        """Get center coordinates for house"""
        r = self.radius
        c = self.center
        offset = r * 0.5
        
        positions = {
            1: (c, c),
            2: (c - offset*0.6, c - r*0.65),
            3: (c - r*0.65, c - offset*0.6),
            4: (c - r*0.65, c),
            5: (c - r*0.65, c + offset*0.6),
            6: (c - offset*0.6, c + r*0.65),
            7: (c, c + r*0.65),
            8: (c + offset*0.6, c + r*0.65),
            9: (c + r*0.65, c + offset*0.6),
            10: (c + r*0.65, c),
            11: (c + r*0.65, c - offset*0.6),
            12: (c + offset*0.6, c - r*0.65)
        }
        
        return positions.get(house, (c, c))
    
    def _draw_house_labels(self, draw: ImageDraw.Draw):
        """Draw house numbers"""
        try:
            font = ImageFont.truetype("arial.ttf", 28)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        for house in range(1, 13):
            x, y = self._get_house_center(house)
            
            # House number
            bbox = draw.textbbox((0, 0), str(house), font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x - w//2, y - h - 25), str(house), 
                     fill='#000080', font=font)
    
    def _draw_planets(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Draw planets in houses"""
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Group by house
        house_planets = {}
        for pos in positions:
            if pos.house not in house_planets:
                house_planets[pos.house] = []
            house_planets[pos.house].append(pos)
        
        for house, planets in house_planets.items():
            x, y = self._get_house_center(house)
            
            for i, pos in enumerate(planets):
                py = y + (i * 28) - 10
                color = PLANET_COLORS[pos.planet]
                
                # Planet text
                text = pos.planet.value[:2]
                if pos.degrees:
                    text += f" {int(pos.degrees)}°"
                
                bbox = draw.textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                
                draw.text((x - w//2, py), text, fill=color, font=font)
    
    def _draw_aspects(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Draw Graha Drishti arrows"""
        for pos in positions:
            aspects = GRAHA_DRISHTI[pos.planet]
            
            for offset in aspects:
                target_house = ((pos.house - 1 + offset) % 12) + 1
                
                x1, y1 = self._get_house_center(pos.house)
                x2, y2 = self._get_house_center(target_house)
                
                color = PLANET_COLORS[pos.planet]
                self._draw_arrow(draw, x1, y1, x2, y2, color)
    
    def _draw_arrow(self, draw: ImageDraw.Draw, x1: int, y1: int,
                    x2: int, y2: int, color: str):
        """Draw curved arrow with transparency"""
        # Convert hex to RGBA
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        rgba = (r, g, b, 100)  # Semi-transparent
        
        # Draw curved line
        draw.line([(x1, y1), (x2, y2)], fill=rgba, width=3)
        
        # Arrow head
        angle = np.arctan2(y2 - y1, x2 - x1)
        arrow_len = 20
        arrow_angle = np.pi / 6
        
        x3 = x2 - arrow_len * np.cos(angle - arrow_angle)
        y3 = y2 - arrow_len * np.sin(angle - arrow_angle)
        x4 = x2 - arrow_len * np.cos(angle + arrow_angle)
        y4 = y2 - arrow_len * np.sin(angle + arrow_angle)
        
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=rgba)

# ==================== INTERPRETATION ====================

class VedicInterpretation:
    """Generate Vedic interpretations"""
    
    @staticmethod
    def generate_dataframe(positions: List[PlanetPosition]) -> pd.DataFrame:
        """Create interpretation table"""
        data = []
        
        for pos in positions:
            aspects = [str((pos.house - 1 + off) % 12 + 1) 
                      for off in GRAHA_DRISHTI[pos.planet]]
            
            data.append({
                'Planet': pos.planet.value,
                'House': pos.house,
                'Degrees': f"{pos.degrees:.1f}°" if pos.degrees else "N/A",
                'Body Part': KALAPURUSHA_BODY_PARTS[pos.house],
                'Life Area': HOUSE_SIGNIFICATIONS[pos.house],
                'Nature': PLANET_NATURE[pos.planet],
                'Aspects Houses': ", ".join(aspects),
                'Gemstone': PLANET_GEMSTONES[pos.planet],
                'Donation Grain': PLANET_GRAINS[pos.planet]
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def health_analysis(positions: List[PlanetPosition]) -> Dict[int, List[str]]:
        """Health impact analysis"""
        impacts = {i: [] for i in range(1, 13)}
        
        for pos in positions:
            nature = "✓ Strengthens" if "Benefic" in PLANET_NATURE[pos.planet] else "⚠ Challenges"
            impacts[pos.house].append(
                f"{nature}: {pos.planet.value} in {KALAPURUSHA_BODY_PARTS[pos.house]}"
            )
        
        for pos in positions:
            for offset in GRAHA_DRISHTI[pos.planet]:
                target = ((pos.house - 1 + offset) % 12) + 1
                nature = "→ Supports" if "Benefic" in PLANET_NATURE[pos.planet] else "→ Stresses"
                impacts[target].append(
                    f"{nature}: {pos.planet.value} aspects from House {pos.house}"
                )
        
        return impacts

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Kalapurusha Vedic Astrology",
        page_icon="🕉️",
        layout="wide"
    )
    
    st.title("🕉️ Kalapurusha Vedic Astrology System")
    st.markdown("**Upload your Lagna Kundali → Get instant Graha Drishti analysis**")
    
    # Sidebar
    with st.sidebar:
        st.header("📖 About")
        st.info("""
        This system analyzes your kundali using:
        - **Kalapurusha** (Cosmic Body mapping)
        - **Graha Drishti** (Classical Vedic aspects)
        - **Health Analysis** (Body part influences)
        
        Upload a clear North Indian diamond chart.
        """)
        
        st.header("🔮 Aspect Rules")
        st.markdown("""
        - **Sun, Moon, Mercury, Venus**: 7th
        - **Mars**: 4th, 7th, 8th
        - **Jupiter**: 5th, 7th, 9th
        - **Saturn**: 3rd, 7th, 10th
        - **Rahu, Ketu**: 5th, 7th, 9th
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your Lagna Kundali (PNG/JPG)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        # Show uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Kundali")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Process button
        if st.button("🔮 Analyze Kundali", type="primary"):
            with st.spinner("Analyzing kundali with Vedic principles..."):
                try:
                    # OCR extraction
                    ocr = EnhancedKundaliOCR()
                    positions = ocr.extract_from_image(image)
                    
                    if not positions:
                        st.error("No planets detected. Please upload a clearer image.")
                        return
                    
                    st.success(f"✓ Detected {len(positions)} planets")
                    
                    # Generate chart
                    chart_gen = KalapurushaChartGenerator(1200)
                    chart_img = chart_gen.generate(positions)
                    
                    with col2:
                        st.subheader("🕉️ Kalapurusha Chart")
                        st.image(chart_img, use_column_width=True)
                    
                    # Interpretation
                    st.divider()
                    st.header("📊 Vedic Analysis")
                    
                    interpreter = VedicInterpretation()
                    df = interpreter.generate_dataframe(positions)
                    
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Health analysis
                    st.divider()
                    st.header("🏥 Kalapurusha Health Analysis")
                    
                    health = interpreter.health_analysis(positions)
                    
                    cols = st.columns(3)
                    for i, (house, impacts) in enumerate(health.items()):
                        if impacts:
                            with cols[i % 3]:
                                st.markdown(f"**House {house}**: {KALAPURUSHA_BODY_PARTS[house]}")
                                for impact in impacts:
                                    st.markdown(f"- {impact}")
                                st.markdown("---")
                    
                    # Download options
                    st.divider()
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Download chart
                        buf = BytesIO()
                        chart_img.save(buf, format='PNG')
                        st.download_button(
                            "⬇️ Download Kalapurusha Chart",
                            buf.getvalue(),
                            "kalapurusha_chart.png",
                            "image/png"
                        )
                    
                    with col_b:
                        # Download CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Analysis Report",
                            csv,
                            "vedic_analysis.csv",
                            "text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error processing kundali: {str(e)}")
                    st.info("Tips: Ensure the image is clear, well-lit, and shows a North Indian diamond chart.")
    
    else:
        # Show example
        st.info("👆 Upload your kundali image to begin analysis")
        
        st.markdown("### ✨ Features")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**🔍 Automatic OCR**")
            st.write("Reads planets and degrees automatically")
        with cols[1]:
            st.markdown("**🎯 Graha Drishti**")
            st.write("Classical Vedic aspects with arrows")
        with cols[2]:
            st.markdown("**💊 Health Analysis**")
            st.write("Body part mapping per Kalapurusha")

if __name__ == "__main__":
    main()
