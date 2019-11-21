<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Story</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link rel="stylesheet" type="text/css" href="./main.css">
</head>
<body>

<div class="container">

    <div class="row justify-content-center mt-4 mb-4">
        <h1 class="display-3">Experiments</h1>
    </div>

    <div class="row mt-4 mb-4 justify-content-center">

        <form action="clean.php" method="get">
            <button type="submit button" class="btn btn-warning">Remove All Empty</button>
        </form>

    </div>

    <table class="table">
        <thead class="thead-dark">
            <tr>
                <th scope="col">#</th>
                <th scope="col">Agent ID</th>
                <th scope="col">Environment</th>
                <th scope="col">Date</th>
                <th scope="col">Maximum Reward</th>
            </tr>
        </thead>

        <tbody>

        <?php
        $dirs = array_filter(glob('experiments/*', GLOB_ONLYDIR), 'is_dir');

        $count = 1;
        foreach ($dirs as $agent_path) {
            $agent_id = explode("/", $agent_path)[1];
            $meta = json_decode(file_get_contents($agent_path . "/meta.json"), true);
            $progress = json_decode(file_get_contents($agent_path . "/progress.json"), true);

            ?><tr>
                <th scope="row"><?php echo $count ?></th>
                <td><a href="experiment.php?id=<?php echo $agent_id ?>"><?php echo $agent_id ?></a></td>
                <td><?php echo $meta["environment"]["name"] ?></td>
                <td><?php echo $meta["date"] ?></td>
                <td><?php echo max($progress["rewards"]) ?></td>
            </tr><?php

            $count++;

        }

        ?>

        </tbody>
    </table>
</div>

</body>
</html>