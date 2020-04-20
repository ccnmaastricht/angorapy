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

function addvector(a, b) {
    return a.map((e, i) => parseFloat(e) + parseFloat(b[i]));
}

function subtractvector(a, b) {
    return a.map((e, i) => parseFloat(e) - parseFloat(b[i]));
}

function sqrtvector(a) {
    return a.map((e) => Math.sqrt(e));
}

// PROGRESS PLOTS
reward_plot_div = document.getElementById('reward-plot');
reward_boxplot_div = document.getElementById('reward-boxplot');
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
let url_elements = $(location).attr("pathname").split("/");
let expid = url_elements[url_elements.length - 1];
console.log(expid);

$.when(
    $.get(Flask.url_for("expfile", {"filename": "meta.json", "exp_id": expid})),
    $.get(Flask.url_for("expfile", {"filename": "progress.json", "exp_id": expid}))
).then(function (req_1, req_2) {
    let meta = req_1[0];
    let prog = req_2[0];

    console.log(meta["environment"]["reward_threshold"]);

    let reward_means = prog["rewards"]["mean"];

    // rewards with slider for smoothing
    let window_sizes = [0, 1, 2, 3, 4];

    let reward_traces = [];
    for (let ws in window_sizes) {
        reward_traces.push({
            x: _.range(_.size(reward_means)),
            y: smooth(reward_means, window_sizes[ws]),
            mode: "lines",
            yaxis: 'y',
            name: "Reward",
            line: {
                color: "red",
                width: 2,
                smoothing: 10
            },
            visible: ws === "0",
        })
    }


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
        x: _.range(_.size(prog["lengths"]["mean"])),
        y: prog["lengths"]["mean"],
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
        update_menus: [{
            buttons: [
                {
                    args: ['shapes', []],
                    label: 'None',
                    method: 'relayout'
                },
                {
                    args: ['shapes', [{
                        type: 'line',
                        x0: 0, y0: meta["environment"]["reward_threshold"],
                        x1: reward_means.length, y1: meta["environment"]["reward_threshold"],
                        layer: "below",
                        line: {
                            color: 'grey',
                            width: 2,
                            dash: 'dashdot',
                        }
                    },]],
                    label: 'Threshold Line',
                    method: 'relayout'
                },
            ],
            direction: 'left',
            pad: {'r': 10, 't': 10},
            showactive: true,
            type: 'buttons',
            x: 0,
            xanchor: 'left',
            y: 0,
            yanchor: 'top'
        },]
    }, standard_layout);

    Plotly.newPlot(reward_plot_div, traces, layout, {responsive: true});

    // REWARD LENGTH BOXPLOTS

    let data = [{
        y: prog["rewards"]["last_cycle"],
        boxpoints: 'all',
        jitter: 0.3,
        pointpos: -1.8,
        type: 'box',
        name: "Reward",
        marker: {color: "Red"}
    }, {
        y: prog["lengths"]["last_cycle"],
        yaxis: 'y2',
        boxpoints: 'all',
        jitter: 0.3,
        pointpos: -1.8,
        type: 'box',
        name: "Length",
        marker: {color: "Orange"}
    }];

    Plotly.newPlot(reward_boxplot_div, data, {
            ...standard_layout,
            title: "Episode Reward and Length Distribution",
            yaxis2: {title: "Episode Steps", side: "right", overlaying: 'y', showgrid: false},

        },
        {responsive: true});


    // OBJECTIVE PLOTS
    let reference_reward_trace = {
        x: _.range(_.size(reward_means)), y: reward_means,
        mode: "lines", name: "Reward", line: {color: "lightgrey", dash: "dot"}, yaxis: 'y2',
    };

    let objective_layout = _.merge({
        yaxis2: {title: "Cumulative Return", side: "right", overlaying: 'y', showgrid: false},
        yaxis: {title: "Loss", showgrid: false},
    }, standard_layout);

    // ENTROPY PLOT
    console.log(prog["entropies"]);
    while (prog["entropies"].indexOf("NaN") > -1) {
        prog["entropies"][prog["entropies"].indexOf("NaN")] = NaN
    }

    let entropy_trace = {
        x: _.range(_.size(prog["entropies"])), y: prog["entropies"],
        mode: "lines", name: "Approximate Entropy",
        marker: {color: "green"},
        yaxis: 'y',
    };

    Plotly.newPlot(entropy_plot_div, [entropy_trace, reference_reward_trace], _.merge({
        title: "Approximate Entropy",
    }, objective_layout), {responsive: true});

    // Policy Loss PLOT
    let ploss_trace = {
        x: _.range(_.size(prog["ploss"])), y: prog["ploss"],
        mode: "lines", name: "Policy Loss",
        marker: {color: "Turquoise"},
    };

    Plotly.newPlot(ploss_plot_div, [ploss_trace, reference_reward_trace], _.merge({
        title: "Policy Loss",
    }, objective_layout), {responsive: true});

    // VALUE LOSS PLOT
    let vloss_trace = {
        x: _.range(_.size(prog["vloss"])), y: prog["vloss"],
        mode: "lines", name: "Value Loss",
        marker: {color: "Tomato "},
    };

    Plotly.newPlot(vloss_plot_div, [vloss_trace, reference_reward_trace], _.merge({
        title: "Value Loss",
    }, objective_layout), {responsive: true});

    // NORMALIZATION PLOTS
    let norm_plot_x = _.range(_.size(prog["preprocessors"]["RewardNormalizationWrapper"]["mean"]))
    let rew_norm_traces = [];
    for (let i = 0; i < prog["preprocessors"]["RewardNormalizationWrapper"]["mean"][0].length; i++) {
        rew_norm_traces.push({
            x: norm_plot_x,
            y: prog["preprocessors"]["RewardNormalizationWrapper"]["mean"].map(x => x[i]),
            mode: "lines",
            name: "Running Mean Reward",
            marker: {color: "Green"},
        });
    }

    let weird_x = _.concat(
        norm_plot_x,
        _.reverse(_.range(_.size(prog["preprocessors"]["RewardNormalizationWrapper"]["mean"]))))

    let upper_reward_bound = addvector(
        prog["preprocessors"]["RewardNormalizationWrapper"]["mean"],
        prog["preprocessors"]["RewardNormalizationWrapper"]["stdev"])

    console.log(upper_reward_bound);

    let lower_reward_bound = subtractvector(
        prog["preprocessors"]["RewardNormalizationWrapper"]["mean"],
        prog["preprocessors"]["RewardNormalizationWrapper"]["stdev"])

    rew_norm_traces.push({
        x: norm_plot_x,
        y: lower_reward_bound,
        line: {color: "transparent"},
        showlegend: false,
        type: "scatter"
    })

    rew_norm_traces.push({
        x: norm_plot_x,
        y: upper_reward_bound,
        fill: "tonexty",
        fillcolor: 'rgba(68, 68, 68, 0.1)',
        line: {color: "transparent"},
        showlegend: false,
        type: "scatter"
    })

    Plotly.newPlot(rew_norm_plot_div, rew_norm_traces, _.merge({
        title: "Reward Normalization",
    }, standard_layout), {responsive: true});

    let state_norm_traces = [];
    for (let i = 0; i < prog["preprocessors"]["StateNormalizationWrapper"]["mean"][0].length; i++) {
        state_norm_traces.push({
            x: _.range(_.size(prog["preprocessors"]["StateNormalizationWrapper"]["mean"])),
            y: prog["preprocessors"]["StateNormalizationWrapper"]["mean"].map(x => x[i]),
            mode: "lines",
            name: "Running Mean State",
            marker: {color: "Green"},
        });
    }

    Plotly.newPlot(state_norm_plot_div, state_norm_traces, _.merge({
        title: "State Normalization",
    }, standard_layout), {responsive: true});
});

