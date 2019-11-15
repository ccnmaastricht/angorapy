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
        <h1>Experiment: <?php echo $meta["environment"] ?></h1>
    </div>

    <div class="row justify-content-center mt-4">
        <div class="col">
            <table class="table table-striped">
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
    </div>

    <div class="row justify-content-center mt-4">
        <div class="col col-6">
            <img src="<?php echo $DIR . "./reward_plot.svg" ?>"/>
        </div>

        <div class="col col-6">
            <img src="<?php echo $DIR . "./loss_plot.svg" ?>"/>
        </div>
    </div>
</div>

</body>
</html>