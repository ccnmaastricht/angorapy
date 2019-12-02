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

    <!-- Bootstrap -->
    <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link rel="stylesheet" type="text/css" href="./main.css">

    <!--  CHART JS  -->
<!--    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.8.0/Chart.min.js"></script>-->

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
            <img src="<?php echo $DIR . "./reward_plot.svg" ?>" alt="Reward Plot" class="fit-div" />
        </div>

        <div class="col col-6">
            <img src="<?php echo $DIR . "./loss_plot.svg" ?>" alt="Loss Plot" class="fit-div" />
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
                    ?><tr>
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
                    ?><tr>
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
                    foreach($gifs as $filename) {
                        $iteration = intval(explode("_", $filename)[1])

                        ?><div class="carousel-item <?php echo ($i == 0 ? 'active' : '')?>">
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
    $('.carousel').carousel()
</script>

</body>
</html>