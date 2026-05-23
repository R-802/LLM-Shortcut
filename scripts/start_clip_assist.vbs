' Start Clip Assist in the background (no console window).
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
root = fso.GetParentFolderName(scriptDir)
launcher = scriptDir & "\run_service.bat"
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = root
sh.Run "cmd /c """ & launcher & """", 0, False
