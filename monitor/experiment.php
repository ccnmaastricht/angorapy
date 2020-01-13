<?php
$DIR = "experiments/" . $_GET["id"] . "/";
$meta = json_decode(file_get_contents($DIR . "/meta.json"), true);
$progress = json_decode(file_get_contents($DIR . "/progress.json"), true);
?>

<!doctype html>
<html lang="en">
<head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!--    Font -->
    <link href="https://fonts.googleapis.com/css?family=Fira+Sans&display=swap" rel="stylesheet">

    <!--  JQUERY  -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/lodash@4.17.15/lodash.min.js"></script>

    <!-- Bootstrap -->
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js"
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo"
            crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6"
            crossorigin="anonymous"></script>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
          integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link rel="stylesheet" type="text/css" href="./main.css">


    <!--  PLOTTING LIBRARY  -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>


    <title>Experiment</title>
</head>

<body>
<div class="container">
    <div class="row justify-content-center">
        <div class="main-title">
            <h1 class="display-3"><?php echo $meta["environment"]["name"] ?><br></h1>
            <h3> <?php echo $_GET["id"]; ?> </h3>
        </div>
    </div>

    <div class="row justify-content-center mt-3">
        <div class="col col-6">
            <div id="reward-plot" style="width:100%; height:500px;"></div>
        </div>

        <div class="col col-6">
            <div id="entropy-plot" style="width:100%; height:500px"></div>
        </div>
    </div>

    <!--  OBJECTIVE PLOTS  -->
    <div class="row justify-content-center mt-3">
        <h5 align="center" class="sub-title">Objectives</h5>
    </div>

    <div class="row justify-content-center">
        <div class="col col-6">
            <div id="ploss-plot" style="width:100%; height:500px"></div>

        </div>

        <div class="col col-6">
            <div id="vloss-plot" style="width:100%; height:500px"></div>
        </div>
    </div>


    <!--  NORMALIZATION PLOTS  -->
    <div class="row justify-content-center mt-3">
        <h5 align="center" class="sub-title">Normalization Preprocessors</h5>
    </div>

    <div class="row justify-content-center mt-3">
        <div class="col col-6">
            <div id="snorm-plot" style="width:100%; height:500px"></div>
        </div>

        <div class="col col-6">
            <div id="rnorm-plot" style="width:100%; height:500px"></div>
        </div>
    </div>


    <div class="row justify-content-center mt-5">
        <div class="col col-6">
            <h4 align="center" class="sub-title">Hyperparameters</h4>

            <table class="table table-striped" title="Hyperparameters">
                <thead>
                <tr>
                    <th scope="col">Hyperparameter</th>
                    <th scope="col">Value</th>
                </tr>
                </thead>
                <tbody>
                <?php
                foreach ($meta["hyperparameters"] as $p => $v) {
                    ?>
                    <tr>
                    <td><?php echo $p ?></td>
                    <td><?php echo $v ?></td>
                    </tr><?php
                }
                ?>
                </tbody>
            </table>
        </div>

        <div class="col col-6">
            <h4 align="center" class="sub-title">Environment</h4>

            <table class="table table-striped" title="Hyperparameters">
                <thead>
                <tr>
                    <th scope="col">Attribute</th>
                    <th scope="col">Value</th>
                </tr>
                </thead>
                <tbody>
                <?php
                foreach ($meta["environment"] as $p => $v) {
                    ?>
                    <tr>
                    <td><?php echo $p ?></td>
                    <td><?php echo $v ?></td>
                    </tr><?php
                }
                ?>
                </tbody>
            </table>
        </div>
    </div>

    <div class="row justify-content-center mt-5">
        <div class="col">
            <h4 align="center" class="sub-title">Episode GIFs</h4>

            <div id="carouselExampleControls" class="carousel slide" data-ride="carousel">
                <div class="carousel-inner">
                    <?php
                    $gifs = glob($DIR . '/*.gif');
                    $i = 0;
                    foreach ($gifs as $filename) {
                        $iteration = intval(explode("_", $filename)[1])

                        ?>
                        <div class="carousel-item <?php echo($i == 0 ? 'active' : '') ?>">
                        <img src="<?php echo $filename ?>" class="d-block w-100" alt="...">
                        <div class="carousel-caption d-none d-md-block">
                            <h5 style="color: black">Iteration <?php echo $iteration ?></h5>
                            <p style="color: dimgray">Average reward of <?php ?></p>
                        </div>
                        </div><?php

                        $i++;
                    }
                    ?>
                </div>
                <a class="carousel-control-prev" href="#carouselExampleControls" role="button" data-slide="prev">
                    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                    <span class="sr-only">Previous</span>
                </a>
                <a class="carousel-control-next" href="#carouselExampleControls" role="button" data-slide="next">
                    <span class="carousel-control-next-icon" aria-hidden="true"></span>
                    <span class="sr-only">Next</span>
                </a>
            </div>
        </div>
    </div>

    <div class="row justify-content-center mt-5">
        <div class="col col-8">
            <h4 align="center" class="sub-title">Model</h4>

            <img src="<?php echo $DIR . '/model.png' ?>" alt="Plot of the Model" class="fit-div"/>
        </div>
    </div>
</div>

<script>
    // GIF CAROUSEL
    $('.carousel').carousel();

    // PROGRESS PLOTS
    reward_plot_div = document.getElementById('reward-plot');
    entropy_plot_div = document.getElementById('entropy-plot');
    vloss_plot_div = document.getElementById('vloss-plot');
    ploss_plot_div = document.getElementById('ploss-plot');

    let standard_layout = {
        margin: {
            l: 50, r: 50, b: 50, t: 50, pad: 4
        },
        showlegend: true,
        legend: {
            orientation: "h",
            xanchor: "center",
            yanchor: "top",
            y:-0.3,
            x:0.5
        }
    };

    $.ajax({
        type: "Get",
        url: "<?php echo $DIR ?>/progress.json",
        dataType: "json",
        success: function (data) {
            console.log(data);

            // REWARD/LENGTH PLOT
            let reward_trace = {
                x: _.range(_.size(data["rewards"])),
                y: data["rewards"],
                mode: "lines",
                yaxis: 'y',
                name: "Reward"
            };

            let length_trace = {
                x: _.range(_.size(data["lengths"])),
                y: data["lengths"],
                mode: "lines",
                name: "Episode Length",
                yaxis: 'y2',
            };

            let traces = [length_trace, reward_trace];
            let layout = {
                title: "Average Rewards and Episode Lengths",
                ...standard_layout,
                yaxis: {title: "Return"},
                yaxis2: {title: "Steps", side: "right", overlaying: 'y',},

            };

            Plotly.newPlot(reward_plot_div, traces, layout);


            // ENTROPY PLOT
            let entropy_trace = {
                x: _.range(_.size(data["entropies"])), y: data["entropies"],
                mode: "lines", name: "Approximate Entropy",
                marker: {color: "green"},
            };

            Plotly.newPlot(entropy_plot_div, [entropy_trace], {
                title: "Approximate Entropy",
                ...standard_layout
            });

            // Policy Loss PLOT
            let ploss_trace = {
                x: _.range(_.size(data["ploss"])), y: data["ploss"],
                mode: "lines", name: "Policy Loss",
                marker: {color: "Turquoise"},
            };

            Plotly.newPlot(ploss_plot_div, [ploss_trace], {
                title: "Policy Loss",
                ...standard_layout
            });

            // VALUE LOSS PLOT
            let vloss_trace = {
                x: _.range(_.size(data["vloss"])), y: data["vloss"],
                mode: "lines", name: "Value Loss",
                marker: {color: "Tomato "},
            };

            Plotly.newPlot(vloss_plot_div, [vloss_trace], {
                title: "Value Loss",
                ...standard_layout
            });
        },

        error: function () {
            alert("NO DATA FOUND");
        }
    });


    // reward_plot_div = document.getElementById('reward-plot');
    // Plotly.plot(reward_plot_div, [{
    //     x: [1, 2, 3, 4, 5],
    //     y: [1, 2, 4, 8, 16]
    // }], {
    //     margin: {t: 0}
    // });
</script>

</body>
</html>