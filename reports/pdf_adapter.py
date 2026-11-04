"""
PDF Generation Adapter Layer - ENHANCED VERSION
Better image handling to fix "Cannot open resource" errors
"""

from typing import Dict, Optional, List
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os
from reports.pdf_report import generate_clinical_pdf


class PDFAdapter:
    """
    Enhanced adapter with robust image handling
    """
    
    @staticmethod
    def generate_report(
        save_path: str,
        analysis_result: Dict,
        original_image: np.ndarray,
        patient_info: Optional[Dict] = None
    ) -> bool:
        """
        Generate PDF report with complete data conversion
        """
        temp_files = []
        
        try:
            print(f"📄 Starting PDF generation...")
            print(f"   Original image shape: {original_image.shape}")
            print(f"   Original image dtype: {original_image.dtype}")
            
            # ═══════════════════════════════════════════════════
            # 1️⃣ Save images to temp files with proper validation
            # ═══════════════════════════════════════════════════
            
            # Save original image
            img_original_path = PDFAdapter._save_temp_image(
                original_image, 
                "original", 
                temp_files
            )
            print(f"   ✓ Original saved: {img_original_path}")
            
            # Save GradCAM
            gradcam_array = analysis_result.get('gradcam')
            if gradcam_array is not None and isinstance(gradcam_array, np.ndarray):
                print(f"   GradCAM shape: {gradcam_array.shape}")
                print(f"   GradCAM dtype: {gradcam_array.dtype}")
                img_gradcam_path = PDFAdapter._save_temp_image(
                    gradcam_array, 
                    "gradcam", 
                    temp_files
                )
                print(f"   ✓ GradCAM saved: {img_gradcam_path}")
            else:
                print(f"   ⚠️  No GradCAM array, using original")
                img_gradcam_path = img_original_path
            
            # Attention map (same as gradcam for now)
            img_attention_path = img_gradcam_path
            
            # Verify files exist
            for path in [img_original_path, img_gradcam_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Temp image file not created: {path}")
                file_size = os.path.getsize(path)
                if file_size == 0:
                    raise ValueError(f"Temp image file is empty: {path}")
                print(f"   ✓ Verified: {path} ({file_size} bytes)")
            
            # ═══════════════════════════════════════════════════
            # 2️⃣ Build regions list
            # ═══════════════════════════════════════════════════
            
            regions = PDFAdapter._build_regions_list(analysis_result, patient_info)
            print(f"   ✓ Built {len(regions)} regions")
            
            # ═══════════════════════════════════════════════════
            # 3️⃣ Convert prediction to int
            # ═══════════════════════════════════════════════════
            
            pred_class = PDFAdapter._convert_prediction_to_int(
                analysis_result['prediction']
            )
            confidence = float(analysis_result['confidence'])
            
            print(f"   ✓ Prediction: {pred_class} ({confidence:.1%})")
            
            # ═══════════════════════════════════════════════════
            # ✅ Call PDF generator
            # ═══════════════════════════════════════════════════
            
            print(f"   📄 Generating PDF: {save_path}")
            
            result = generate_clinical_pdf(
                pdf_path=save_path,
                img_original=img_original_path,
                img_gradcam=img_gradcam_path,
                img_attention=img_attention_path,
                pred_class=pred_class,
                confidence=confidence,
                regions=regions
            )
            
            print(f"   ✅ PDF generation completed!")
            return result
            
        except Exception as e:
            print(f"   ❌ Error in PDF adapter: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            # Clean up temp files
            PDFAdapter._cleanup_temp_files(temp_files)
    
    @staticmethod
    def _save_temp_image(
        img_array: np.ndarray, 
        prefix: str,
        temp_files: List[str]
    ) -> str:
        """
        Save numpy array as temporary image with validation
        """
        if img_array is None:
            raise ValueError(f"Image array is None for {prefix}")
        
        if not isinstance(img_array, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(img_array)}")
        
        if img_array.size == 0:
            raise ValueError(f"Image array is empty for {prefix}")
        
        # Ensure image is in correct format
        if len(img_array.shape) == 2:
            # Grayscale - convert to BGR
            img_to_save = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 3:
                # Already BGR or RGB - cv2 expects BGR
                # If it's RGB (from matplotlib/PIL), convert to BGR
                # For safety, we'll assume it might be either
                img_to_save = img_array.copy()
            elif img_array.shape[2] == 4:
                # RGBA - convert to BGR
                img_to_save = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                raise ValueError(f"Unexpected number of channels: {img_array.shape[2]}")
        else:
            raise ValueError(f"Unexpected image shape: {img_array.shape}")
        
        # Ensure uint8 format
        if img_to_save.dtype != np.uint8:
            # Normalize to 0-255 if needed
            if img_to_save.max() <= 1.0:
                img_to_save = (img_to_save * 255).astype(np.uint8)
            else:
                img_to_save = img_to_save.astype(np.uint8)
        
        # Create temp file with explicit directory
        temp_dir = tempfile.gettempdir()
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.png', 
            prefix=f'{prefix}_',
            dir=temp_dir
        )
        os.close(temp_fd)
        
        # Save with cv2
        success = cv2.imwrite(temp_path, img_to_save)
        
        if not success:
            raise IOError(f"Failed to write image to {temp_path}")
        
        # Verify file was created and has content
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temp file was not created: {temp_path}")
        
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise ValueError(f"Temp file is empty: {temp_path}")
        
        temp_files.append(temp_path)
        return temp_path
    
    @staticmethod
    def _convert_prediction_to_int(prediction: str) -> int:
        """Convert prediction from string to int"""
        if isinstance(prediction, int):
            return prediction
        
        if isinstance(prediction, str):
            pred_upper = prediction.upper()
            if pred_upper in ['MALIGNANT', 'CANCER', 'POSITIVE', '1']:
                return 1
            elif pred_upper in ['BENIGN', 'NORMAL', 'NEGATIVE', '0']:
                return 0
            else:
                raise ValueError(f"Unknown prediction value: {prediction}")
        
        raise TypeError(f"Prediction must be str or int, got {type(prediction)}")
    
    @staticmethod
    def _build_regions_list(
        analysis_result: Dict,
        patient_info: Optional[Dict] = None
    ) -> List[Dict]:
        """Build regions list with proper format"""
        features = analysis_result.get('features', {})
        expert = analysis_result.get('expert_analysis', {})
        
        # Create single region
        region = {
            'center': features.get('center', (128, 128)),
            'radius': float(features.get('radius', features.get('size', 10.0))),
            'size_pixels': int(features.get('size_pixels', features.get('size', 100))),
            'malignancy_score': float(analysis_result.get('confidence', 0.0)),
            
            'features': {
                'size_mm': features.get('size_mm', features.get('size')),
                'sphericity': features.get('sphericity', 0.8),
                'margin': features.get('margin', 'smooth'),
                'texture': features.get('texture', 'homogeneous'),
                'calcification': features.get('calcification', False),
                'spiculation': features.get('spiculation', 'absent'),
            },
            
            'expert_analysis': {
                'risk_level': expert.get('risk_level', 'UNKNOWN'),
                'recommendation': expert.get('recommendation', 'Further evaluation needed'),
                'factors': expert.get('factors', []),
            },
            
            'prediction': analysis_result.get('prediction', 'UNKNOWN'),
            'confidence': float(analysis_result.get('confidence', 0.0)),
        }
        
        if patient_info:
            region['patient_info'] = patient_info
        
        return [region]
    
    @staticmethod
    def _cleanup_temp_files(temp_files: List[str]):
        """Clean up temporary files"""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"   🗑️  Cleaned up: {temp_file}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not delete {temp_file}: {e}")


# ═══════════════════════════════════════════════════════════════
# Convenience Function
# ═══════════════════════════════════════════════════════════════

def generate_pdf_report(
    save_path: str,
    analysis_result: Dict,
    original_image: np.ndarray,
    patient_info: Optional[Dict] = None
) -> bool:
    """
    Convenience function for GUI usage
    """
    return PDFAdapter.generate_report(
        save_path, 
        analysis_result, 
        original_image, 
        patient_info
    )


if __name__ == "__main__":
    print("✅ Enhanced PDF Adapter Module")
    print("="*60)
    print("Improvements:")
    print("  ✅ Better image format handling")
    print("  ✅ Explicit dtype conversion to uint8")
    print("  ✅ File validation after creation")
    print("  ✅ Detailed error logging")
    print("="*60)