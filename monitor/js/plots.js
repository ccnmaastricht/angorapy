let url_params = new URLSearchParams(window.location.search);
let exp_dir = "experiments/" + url_params.get("id");

// PROGRESS PLOTS
reward_plot_div = document.getElementById('reward-plot');
entropy_plot_div = document.getElementById('entropy-plot');

vloss_plot_div = document.getElementById('vloss-plot');
ploss_plot_div = document.getElementById('ploss-plot');

rew_norm_plot_div = document.getElementById('rew-norm-plot');
state_norm_plot_div = document.getElementById('state-norm-plot');

let standard_layout = {
    margin: {
        l: 50, r: 50, b: 50, t: 50, pad: 4
    },
    showlegend: true,
    legend: {
        orientation: "h",
        xanchor: "center",
        yanchor: "top",
        y: -0.3,
        x: 0.5
    },
    xaxis: {
        zeroline: false,
        showline: true,
        showgrid: false,
        mirror: 'ticks',
    },
    yaxis: {
        zeroline: false,
        showline: true,
        showgrid: false,
        mirror: 'ticks',
    }
};

$.ajax({
    type: "Get",
    url: exp_dir + "/progress.json",
    dataType: "json",
    success: function (data) {
        console.log(data);

        // REWARD/LENGTH PLOT
        let reward_trace = {
            x: _.range(_.size(data["rewards"]["mean"])),
            y: data["rewards"]["mean"],
            mode: "lines",
            yaxis: 'y',
            name: "Reward"
        };

        let length_trace = {
            x: _.range(_.size(data["lengths"]["mean"])),
            y: data["lengths"]["mean"],
            mode: "lines",
            name: "Episode Length",
            yaxis: 'y2',
        };

        let traces = [length_trace, reward_trace];
        let layout = _.merge({
            title: "Average Rewards and Episode Lengths",
            yaxis: {title: "Return"},
            yaxis2: {title: "Steps", side: "right", overlaying: 'y', showgrid: false},
            // shape: [{
            //     type: 'line',
            //     x0: 2,
            //     y0: 2,
            //     x1: 5,
            //     y1: 2,
            //     line: {
            //         color: 'rgb(50, 171, 96)',
            //         width: 4,
            //         dash: 'dashdot'
            //     }
            // }]
        }, standard_layout);

        Plotly.newPlot(reward_plot_div, traces, layout);


        // ENTROPY PLOT
        let entropy_trace = {
            x: _.range(_.size(data["entropies"])), y: data["entropies"],
            mode: "lines", name: "Approximate Entropy",
            marker: {color: "green"},
        };

        Plotly.newPlot(entropy_plot_div, [entropy_trace], _.merge({
            title: "Approximate Entropy",
        }, standard_layout));

        // Policy Loss PLOT
        let ploss_trace = {
            x: _.range(_.size(data["ploss"])), y: data["ploss"],
            mode: "lines", name: "Policy Loss",
            marker: {color: "Turquoise"},
        };

        Plotly.newPlot(ploss_plot_div, [ploss_trace], _.merge({
            title: "Policy Loss",
        }, standard_layout));

        // VALUE LOSS PLOT
        let vloss_trace = {
            x: _.range(_.size(data["vloss"])), y: data["vloss"],
            mode: "lines", name: "Value Loss",
            marker: {color: "Tomato "},
        };

        Plotly.newPlot(vloss_plot_div, [vloss_trace], _.merge({
            title: "Value Loss",
        }, standard_layout));

        // NORMALIZATION PLOTS
        let rew_norm_trace = {
            x: _.range(_.size(data["preprocessors"]["RewardNormalizationWrapper"]["mean"])),
            y: data["preprocessors"]["RewardNormalizationWrapper"]["mean"],
            mode: "lines",
            name: "Running Mean Reward",
            marker: {color: "Green "},
        };

        Plotly.newPlot(rew_norm_plot_div, [rew_norm_trace], _.merge({
            title: "Reward Normalization",
        }, standard_layout));

        let state_norm_trace = {
            x: _.range(_.size(data["preprocessors"]["StateNormalizationWrapper"]["mean"])),
            y: data["preprocessors"]["StateNormalizationWrapper"]["mean"],
            mode: "lines",
            name: "Running Mean State",
            marker: {color: "Green "},
        };

        Plotly.newPlot(state_norm_plot_div, [state_norm_trace], _.merge({
            title: "State Normalization",
        }, standard_layout));
    },

    error: function () {
        alert("NO DATA FOUND");
    }
});