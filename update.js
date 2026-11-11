module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      message: [
        "if exist native\\qwentts-cpp\\.git git -C native\\qwentts-cpp fetch --depth 1 origin de2ad83",
        "if exist native\\qwentts-cpp\\.git git -C native\\qwentts-cpp checkout --force de2ad83",
        "if exist native\\qwentts-cpp\\.git git -C native\\qwentts-cpp submodule update --init --recursive"
      ]
    }
  }, {
    method: "script.start",
    params: {
      uri: "install.js"
    }
  }]
}
