echo "Now installing data from Kaggle."

!pip install -q kaggle

!mkdir ~/.kaggle
! mv kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
!unzip -qq gtsrb-german-traffic-sign.zip