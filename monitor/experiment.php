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
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link rel="stylesheet" type="text/css" href="./main.css">

    <!--  CHART JS  -->
<!--    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.8.0/Chart.min.js"></script>-->

    <title>Experiment</title>
</head>

<body>
<div class="container">
    <div class="row justify-content-center mt-6">
        <h1 class="main-title display-3">Experiment: <?php echo $meta["environment"] ?></h1>
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
            <h4 align="center" class="sub-title">Model</h4>

            <img src="<?php echo $DIR . '/model.png' ?>" alt="Plot of the Model" class="fit-div"/>
        </div>
    </div>

    <div class="row justify-content-center mt-5">
        <div class="col col-6">
            <img src="<?php echo $DIR . "./reward_plot.svg" ?>" alt="Reward Plot" class="fit-div" />
        </div>

        <div class="col col-6">
            <img src="<?php echo $DIR . "./loss_plot.svg" ?>" alt="Loss Plot" class="fit-div" />
        </div>
    </div>
</div>

</body>
</html>