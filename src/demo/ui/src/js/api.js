var API = {

    api_prefix: 'http://' + window.location.hostname + ':8000/',

    get_img_url: function (name) {
        let url = this.api_prefix + 'image/' + name;
        return url;
    },

    get_sample: function (problem, split, idx) {
        return {
            method: "get",
            url: this.api_prefix + 'sample/' + problem + "/" + split + '/' + idx,
        }
    },

    get_explist: function () {
        return {
            method: "get",
            url: this.api_prefix + 'explist',
        }
    },

    get_exps: function (exp_names, idx) {
        return {
            method: "post",
            url: this.api_prefix + 'exps?idx=' + idx,
            data: exp_names
        }
    }
};

export { API };
