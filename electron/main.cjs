const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');

const IMAGE_EXT = new Set(['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp']);
const isDev = process.env.NODE_ENV === 'development';
let mainWindow = null;
let backendProcess = null;

const BACKEND_PORT = 8765;
const FRONTEND_URL = isDev ? 'http://127.0.0.1:5173' : `file://${path.join(__dirname, '../frontend/dist/index.html')}`;

function startBackend() {
  if (isDev) return;
  const python = process.platform === 'win32' ? 'python' : 'python3';
  const backendDir = path.join(__dirname, '../backend');
  backendProcess = spawn(python, ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', `--port`, String(BACKEND_PORT)], {
    cwd: backendDir,
    stdio: 'ignore',
    detached: false,
  });
}

async function scanFolder(dir, recursive = true) {
  const results = [];
  async function walk(current) {
    let entries;
    try {
      entries = await fs.readdir(current, { withFileTypes: true });
    } catch {
      return;
    }
    for (const ent of entries) {
      const full = path.join(current, ent.name);
      if (ent.isDirectory()) {
        if (recursive) await walk(full);
      } else if (ent.isFile()) {
        const ext = path.extname(ent.name).toLowerCase();
        if (IMAGE_EXT.has(ext)) {
          results.push({
            path: full,
            name: ent.name,
            folder: current,
            ext: ext.slice(1),
          });
        }
      }
    }
  }
  await walk(dir);
  return results;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 960,
    minWidth: 1200,
    minHeight: 700,
    backgroundColor: '#0D1117',
    title: 'Annotra',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.loadURL(FRONTEND_URL);
  // if (isDev) mainWindow.webContents.openDevTools({ mode: 'detach' });
}

app.whenReady().then(() => {
  startBackend();
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (backendProcess) backendProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});

ipcMain.handle('dialog:openFolder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Select image dataset folder',
  });
  if (result.canceled || !result.filePaths.length) return null;
  return path.resolve(result.filePaths[0]);
});

ipcMain.handle('dialog:openFile', async (_, filters) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: filters || [{ name: 'All', extensions: ['*'] }],
  });
  if (result.canceled || !result.filePaths.length) return null;
  return result.filePaths[0];
});

ipcMain.handle('dialog:saveFolder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory', 'createDirectory'],
    title: 'Select export destination',
  });
  if (result.canceled || !result.filePaths.length) return null;
  return result.filePaths[0];
});

ipcMain.handle('path:joinDataset', async (_, root, rel) => {
  if (!root || !rel) return null;
  const normalized = rel.replace(/\//g, path.sep);
  const full = path.resolve(path.join(root, normalized));
  const rootResolved = path.resolve(root);
  const relCheck = path.relative(rootResolved, full);
  if (relCheck.startsWith('..') || path.isAbsolute(relCheck)) {
    throw new Error('Invalid dataset path');
  }
  return full;
});

ipcMain.handle('fs:scanFolder', async (_, folderPath, recursive = true) => {
  return scanFolder(folderPath, recursive);
});

ipcMain.handle('fs:readFileBase64', async (_, filePath) => {
  const buf = await fs.readFile(filePath);
  const ext = path.extname(filePath).toLowerCase().replace('.', '');
  const mime = ext === 'jpg' ? 'jpeg' : ext;
  return `data:image/${mime};base64,${buf.toString('base64')}`;
});

ipcMain.handle('fs:readFileBuffer', async (_, filePath) => {
  return fs.readFile(filePath);
});

ipcMain.handle('fs:writeFile', async (_, filePath, data) => {
  await fs.writeFile(filePath, data);
  return true;
});

ipcMain.handle('fs:exists', async (_, filePath) => {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
});

ipcMain.handle('fs:mkdir', async (_, dirPath) => {
  await fs.mkdir(dirPath, { recursive: true });
  return true;
});

ipcMain.handle('shell:showItemInFolder', async (_, filePath) => {
  shell.showItemInFolder(filePath);
});

ipcMain.handle('app:getPaths', () => ({
  userData: app.getPath('userData'),
  documents: app.getPath('documents'),
  backendUrl: `http://127.0.0.1:${BACKEND_PORT}`,
}));
