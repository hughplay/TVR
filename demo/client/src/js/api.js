var API = {
  api: 'http://'+ window.location.hostname +':7000/api/',
  api_default_datadirs: function() {
    return this.api + 'default_datadirs';
  },
  api_default_configdir: function() {
    return this.api + 'default_configdir';
  },
  api_infolist: function(datadir) {
    return this.api + 'infolist?dir=' + datadir;
  },
  api_problems: function() {
    return this.api + 'problems';
  },
  api_random_info: function(problem, user) {
    return this.api + 'random_info?problem=' + problem + '&user=' + user;
  },
  api_submit_test: function() {
    return this.api + 'submit_test';
  },
  api_test_history: function(user, problem) {
    return this.api + 'test_history?user=' + user + '&problem=' + problem;
  },
  api_info: function(infopath) {
    return this.api + 'info?path=' + infopath;
  },
  api_image: function(datadir, imagename) {
    return this.api + 'image?path=' + datadir + '/image/' + imagename;
  },
  api_configs: function(configdir) {
    return this.api + 'configs?dir=' + configdir;
  },
  api_result: function(configpath) {
    return this.api + 'result?config=' + configpath;
  },
  api_test_sample: function(configpath, id) {
    return this.api + 'test_sample?config=' + configpath + '&id=' + id;
  },
  api_h5image: function(configpath, name) {
    return this.api + 'h5image?config=' + configpath + '&image=' + name;
  },
};

export { API };
