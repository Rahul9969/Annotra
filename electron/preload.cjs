const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('marineAPI', {
  joinDatasetPath: (root, rel) => ipcRenderer.invoke('path:joinDataset', root, rel),
  openFolder: () => ipcRenderer.invoke('dialog:openFolder'),
  openFile: (filters) => ipcRenderer.invoke('dialog:openFile', filters),
  saveFolder: () => ipcRenderer.invoke('dialog:saveFolder'),
  scanFolder: (path, recursive) => ipcRenderer.invoke('fs:scanFolder', path, recursive),
  readFileBase64: (path) => ipcRenderer.invoke('fs:readFileBase64', path),
  readFileBuffer: (path) => ipcRenderer.invoke('fs:readFileBuffer', path),
  writeFile: (path, data) => ipcRenderer.invoke('fs:writeFile', path, data),
  exists: (path) => ipcRenderer.invoke('fs:exists', path),
  mkdir: (path) => ipcRenderer.invoke('fs:mkdir', path),
  showItemInFolder: (path) => ipcRenderer.invoke('shell:showItemInFolder', path),
  getPaths: () => ipcRenderer.invoke('app:getPaths'),
});
