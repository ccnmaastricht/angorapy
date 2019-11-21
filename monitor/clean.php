<?php
/* Clean storage from zero progress experiments */
ob_start();
$exit = "NONE";
passthru('python delete_experiments.py', $exit);
$output = ob_get_clean();
var_dump($output);
var_dump($exit);

//http_redirect("../");
