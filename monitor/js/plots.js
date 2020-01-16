// UTILITIES

function smooth(arr, windowSize, getter = (value) => value, setter) {
    const get = getter;
    const result = [];

    for (let i = 0; i < arr.length; i += 1) {
        const leftOffeset = i - windowSize;
        const from = leftOffeset >= 0 ? leftOffeset : 0;
        const to = i + windowSize + 1;

        let count = 0;
        let sum = 0;
        for (let j = from; j < to && j < arr.length; j += 1) {
            sum += get(arr[j]);
            count += 1
        }

        result[i] = setter ? setter(arr[i], sum / count) : sum / count
    }

    return result
}

// EXPERIMENT LOCATION
let url_params = new URLSearchParams(window.location.search);
let exp_dir = "experiments/" + url_params.get("id");

// PROGRESS PLOTS
reward_plot_div = document.getElementById('reward-plot');
entropy_plot_div = document.getElementById('entropy-plot');

vloss_plot_div = document.getElementById('vloss-plot');
ploss_plot_div = document.getElementById('ploss-plot');

rew_norm_plot_div = document.getElementById('rew-norm-plot');
state_norm_plot_div = document.getElementById('state-norm-plot');

// LAYOUT
let standard_layout = {
    margin: {
        l: 50, r: 50, b: 50, t: 50, pad: 4
    },
    showlegend: true,
    legend: {
        orientation: "h",
        xanchor: "center",
        yanchor: "top",
        y: -0.1,
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
    },
};

// READ DATA AND DRAW PLOTS
$.ajax({
    type: "Get",
    url: exp_dir + "/progress.json",
    dataType: "json",
    success: function (data) {

        // rewards with slider for smoothing
        let window_sizes = [0, 1, 2, 3, 4];

        let reward_traces = [];
        for (let ws in window_sizes) {
            reward_traces.push({
                x: smooth(_.range(_.size(data["rewards"]["mean"])), window_sizes[ws]),
                y: smooth(data["rewards"]["mean"], window_sizes[ws]),
                mode: "lines",
                yaxis: 'y',
                name: "Reward",
                line: {
                    color: "red",
                    width: 2,
                    smoothing: 10
                },
                visible: ws === "0",
            })}

        let reward_slider_steps = [];
        let i = 0;
        for (let ws in window_sizes) {
            reward_slider_steps.push({
                label: window_sizes[ws],
                method: 'update',
                args: [{
                    "visible": _.range(_.size(window_sizes)).map(x => x === i).concat([true])
                }]
            });
            i++;
        }


        let length_trace = {
            x: _.range(_.size(data["lengths"]["mean"])),
            y: data["lengths"]["mean"],
            mode: "lines",
            name: "Episode Length",
            yaxis: 'y2',
            line: {
                color: "Orange",
                dash: "dash",
                width: 2
            }
        };

        let traces = [...reward_traces, length_trace];
        let layout = _.merge({
            title: "Average Rewards and Episode Lengths",
            yaxis: {title: "Return"},
            yaxis2: {title: "Steps", side: "right", overlaying: 'y', showgrid: false},
            sliders: [{
                pad: {
                    t: 30
                },
                currentvalue: {
                    xanchor: 'right',
                    prefix: 'Smoothing: ',
                    font: {
                        color: '#888',
                        size: 10
                    }
                },
                steps: reward_slider_steps
            }],
        }, standard_layout);

        Plotly.newPlot(reward_plot_div, traces, layout, {responsive: true});


        // ENTROPY PLOT
        let entropy_trace = {
            x: _.range(_.size(data["entropies"])), y: data["entropies"],
            mode: "lines", name: "Approximate Entropy",
            marker: {color: "green"},
        };

        Plotly.newPlot(entropy_plot_div, [entropy_trace], _.merge({
            title: "Approximate Entropy",
        }, standard_layout), {responsive: true});

        // Policy Loss PLOT
        let ploss_trace = {
            x: _.range(_.size(data["ploss"])), y: data["ploss"],
            mode: "lines", name: "Policy Loss",
            marker: {color: "Turquoise"},
        };

        Plotly.newPlot(ploss_plot_div, [ploss_trace], _.merge({
            title: "Policy Loss",
        }, standard_layout), {responsive: true});

        // VALUE LOSS PLOT
        let vloss_trace = {
            x: _.range(_.size(data["vloss"])), y: data["vloss"],
            mode: "lines", name: "Value Loss",
            marker: {color: "Tomato "},
        };

        Plotly.newPlot(vloss_plot_div, [vloss_trace], _.merge({
            title: "Value Loss",
        }, standard_layout), {responsive: true});

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
        }, standard_layout), {responsive: true});

        let state_norm_trace = {
            x: _.range(_.size(data["preprocessors"]["StateNormalizationWrapper"]["mean"])),
            y: data["preprocessors"]["StateNormalizationWrapper"]["mean"],
            mode: "lines",
            name: "Running Mean State",
            marker: {color: "Green "},
        };

        Plotly.newPlot(state_norm_plot_div, [state_norm_trace], _.merge({
            title: "State Normalization",
        }, standard_layout), {responsive: true});
    },

    error: function () {
        alert("NO DATA FOUND");
    }
});