"""
Generates a Windows Desktop and Project Shortcut (.lnk) with the custom medical icon.
"""

import os
import sys
from pathlib import Path

def create_shortcuts():
    project_dir = Path(__file__).resolve().parent.parent
    icon_path = project_dir / "assets" / "nodule_ai.ico"
    target_bat = project_dir / "start.bat"
    
    # 1. Shortcut in Project Directory
    project_lnk = project_dir / "Lung Nodule AI System.lnk"
    
    # 2. Shortcut on User Desktop
    desktop_dir = Path(os.environ.get("USERPROFILE", "")) / "Desktop"
    desktop_lnk = desktop_dir / "Lung Nodule AI System.lnk"
    
    vbs_script = f"""
Set WshShell = CreateObject("WScript.Shell")
Set Shortcut1 = WshShell.CreateShortcut("{str(project_lnk)}")
Shortcut1.TargetPath = "{str(target_bat)}"
Shortcut1.WorkingDirectory = "{str(project_dir)}"
Shortcut1.IconLocation = "{str(icon_path)},0"
Shortcut1.Description = "Pulmonary Nodule Diagnostic Intelligence System"
Shortcut1.Save

If WshShell.ExpandEnvironmentStrings("%USERPROFILE%") <> "" Then
    Set Shortcut2 = WshShell.CreateShortcut("{str(desktop_lnk)}")
    Shortcut2.TargetPath = "{str(target_bat)}"
    Shortcut2.WorkingDirectory = "{str(project_dir)}"
    Shortcut2.IconLocation = "{str(icon_path)},0"
    Shortcut2.Description = "Pulmonary Nodule Diagnostic Intelligence System"
    Shortcut2.Save
End If
"""
    vbs_path = project_dir / "temp_shortcut.vbs"
    with open(vbs_path, "w", encoding="utf-8") as f:
        f.write(vbs_script)
    
    os.system(f'cscript //nologo "{vbs_path}"')
    if vbs_path.exists():
        vbs_path.unlink()
        
    print(f"Created shortcut in project: {project_lnk}")
    if desktop_lnk.parent.exists():
        print(f"Created shortcut on Desktop: {desktop_lnk}")

if __name__ == "__main__":
    create_shortcuts()

