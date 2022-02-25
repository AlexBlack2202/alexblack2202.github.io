onmessage = (event) => {
  importScripts('/js/amlich-hnd.js');
  setOutputSize("small");
  postMessage( printSelectedMonth());
};