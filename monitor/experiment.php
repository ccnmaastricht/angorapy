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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">

    <title>Experiment</title>

    <!-- Font -->
    <link href="https://fonts.googleapis.com/css?family=Fira+Sans&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.11.2/css/all.css"
          integrity="sha256-46qynGAkLSFpVbEBog43gvNhfrOj+BmwXdxFgVK/Kvc=" crossorigin="anonymous"/>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.11.2/js/all.js"
            integrity="sha256-2JRzNxMJiS0aHOJjG+liqsEOuBb6++9cY4dSOyiijX4=" crossorigin="anonymous"></script>

    <!-- Bootstrap -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
          integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/malihu-custom-scrollbar-plugin/3.1.5/jquery.mCustomScrollbar.min.css">

    <!-- CSS -->
    <link rel="stylesheet" type="text/css" href="./main.css">

    <!--  PLOTTING LIBRARY  -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>

<body>

<div class="wrapper">
    <!-- Sidebar -->
    <nav id="sidebar">
        <div class="sidebar-header">
            <h3><i class="fas fa-flask"></i></h3>
            <h3>Experiment Monitor</h3>
        </div>

        <div class="reload-button">
            <button type="button" id="refresh" class="btn btn-circle btn-md">
                <i class="fas fa-sync"></i>
            </button>
        </div>

        <ul class="list-unstyled components">
            <li class="active"><a href="#statistics-view">Training Statistics</a></li>
            <li><a href="#hp-view">Hyperparameters</a></li>
            <li><a href="#episode-gifs-view">Episode GIFs</a></li>
            <li><a href="#model-view">Network Graph</a></li>
        </ul>

        <ul class="list-unstyled CTAs">
            <li>
                <a href="index.php" class="button-control">
                    <i class="fas fa-list"></i> Overview
                </a>
            </li>
            <li>
                <a href="https://github.com/ccnmaastricht/dexterous-robot-hand" class="button-control" target="_blank">
                    <i class="fab fa-github"></i> Repository
                </a>
            </li>
        </ul>

    </nav>

    <!-- Page Content -->
    <div id="content">
        <div class="main-title">
            <div class="justify-content-center">
                <h1 class="display-3"><?php echo $meta["environment"]["name"] ?><br></h1>
                <h3> <?php echo $_GET["id"]; ?> </h3>
            </div>
        </div>

        <!--        <nav class="navbar navbar-expand-lg navbar-light bg-light">-->
        <!--            <div class="container-fluid">-->
        <!---->
        <!--                <button type="button" id="sidebarCollapse" class="btn btn-info">-->
        <!--                    <i class="fas fa-align-left"></i>-->
        <!--                    <span>Toggle Sidebar</span>-->
        <!--                </button>-->
        <!--            </div>-->
        <!--        </nav>-->

        <div class="topic-group" id="statistics-view">
            <div class="row justify-content-center">
                <div class="col col-6">
                    <div id="reward-plot" style="width:100%; height:500px;"></div>
                </div>
            </div>

            <!--  OBJECTIVE PLOTS  -->
            <div class="row justify-content-center mt-3">
                <h5 align="center" class="sub-title">Objectives</h5>
            </div>

            <div class="row justify-content-center">
                <div class="col col-4">
                    <div id="ploss-plot" style="width:100%; height:500px"></div>

                </div>

                <div class="col col-4">
                    <div id="vloss-plot" style="width:100%; height:500px"></div>
                </div>

                <div class="col col-4">
                    <div id="entropy-plot" style="width:100%; height:500px"></div>
                </div>
            </div>


            <!--  NORMALIZATION PLOTS  -->
            <div class="row justify-content-center mt-3">
                <h5 align="center" class="sub-title">Normalization Preprocessors</h5>
            </div>

            <div class="row justify-content-center mt-3">
                <div class="col col-6">
                    <div id="state-norm-plot" style="width:100%; height:500px"></div>
                </div>

                <div class="col col-6">
                    <div id="rew-norm-plot" style="width:100%; height:500px"></div>
                </div>
            </div>
        </div>

        <div class="topic-group" id="hp-view">
            <div class="row justify-content-center">
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
        </div>

        <div class="topic-group" id="episode-gifs-view">
            <div class="row justify-content-center">
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
                        <a class="carousel-control-prev" href="#carouselExampleControls" role="button"
                           data-slide="prev">
                            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                            <span class="sr-only">Previous</span>
                        </a>
                        <a class="carousel-control-next" href="#carouselExampleControls" role="button"
                           data-slide="next">
                            <span class="carousel-control-next-icon" aria-hidden="true"></span>
                            <span class="sr-only">Next</span>
                        </a>
                    </div>
                </div>
            </div>
        </div>

        <div class="topic-group" id="model-view">
            <div class="row justify-content-center">
                <div class="col col-8">
                    <h4 align="center" class="sub-title">Model</h4>

                    <img src="<?php echo $DIR . '/model.png' ?>" alt="Plot of the Model" class="fit-div"/>
                </div>
            </div>
        </div>

    </div>
</div>
<!--JQUERY-->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
<!--POPPER-->
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"
        integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1"
        crossorigin="anonymous"></script>
<!--BOOTSTRAP-->
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"
        integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6"
        crossorigin="anonymous"></script>
<!--CUSTOM SCROLLBAR-->
<script src="https://cdnjs.cloudflare.com/ajax/libs/malihu-custom-scrollbar-plugin/3.1.5/jquery.mCustomScrollbar.concat.min.js"></script>
<!--LODASH-->
<script src="https://cdn.jsdelivr.net/npm/lodash@4.17.15/lodash.min.js"></script>


<script>
    $(document).ready(function () {
        $("#sidebar").mCustomScrollbar({theme: "minimal"});
        $('#sidebarCollapse').on('click', function () {
            $('#sidebar').toggleClass('active');
            $('.collapse.in').toggleClass('in');
            $('a[aria-expanded=true]').attr('aria-expanded', 'false');
        });
    });

    $('.carousel').carousel();

    $("nav ul li").on("click", function () {
        $("nav").find(".active").removeClass("active");
        $(this).addClass("active");
    });

    $(".reload-button").on("click", function () {
        console.log("jo");
        location.reload();
    });
</script>

<script src="js/plots.js"></script>

</body>
</html>