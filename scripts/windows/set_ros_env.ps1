$env:ROS_DOMAIN_ID = "42"
$env:RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"
$env:ROS_LOCALHOST_ONLY = "0"

Write-Host "ROS_DOMAIN_ID=$env:ROS_DOMAIN_ID"
Write-Host "RMW_IMPLEMENTATION=$env:RMW_IMPLEMENTATION"
Write-Host "ROS_LOCALHOST_ONLY=$env:ROS_LOCALHOST_ONLY"
Write-Host "Use the same values inside WSL so DDS discovery can cross the Windows/WSL boundary."
