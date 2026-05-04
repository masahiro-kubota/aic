# Support Email Draft: Cloud Evaluation Ends Without Valid Output

## Subject

Team masapon: Qualification submission finishes without valid output despite passing local verification

## Body

Dear AI for Industry Challenge Team,

Thank you for organizing the AI for Industry Challenge and for supporting participants throughout the competition.

I am writing regarding a repeated issue with our qualification submissions for team masapon.

Across multiple recent qualification submissions, including the most recent image below, we have observed the same issue. Although each submission image passes our local verification flow before submission, the cloud evaluation appears to finish without producing a valid output. In the portal, the submission ends as "Failed", the "Result file" is empty, and the "Stdout file" only contains the execution header.

The most recent example is:

- Team: masapon
- Submission ID: 679
- Run ID: 072e3040-e7d7-40c0-83be-e4b09b781a7a
- Submitted image URI:
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/masapon:submission-safe-v7-e7bba00-20260505-024538

For this run, the "Stdout file" shows only:

AI for Industry Challenge - 2026
------
Date: 2026-05-04T18:00:18.474058+00:00
Team: N/A
Submission: 679
Run ID: 072e3040-e7d7-40c0-83be-e4b09b781a7a
------

The "Result file" is empty.

Before each submission, we verified the corresponding submission image locally using the standard Docker-based flow:

docker compose \
  -f docker/docker-compose.yaml \
  -f docker/docker-compose.submission_safe_v7.yaml \
  up

I have attached the log from the command above as "local_verification.log".

Compose files used for local verification:
https://github.com/masahiro-kubota/aic/blob/5-4/docker/docker-compose.yaml
https://github.com/masahiro-kubota/aic/blob/5-4/docker/docker-compose.submission_safe_v7.yaml
The override file only changes the docker image name for the model service.

Could you please check the cluster-side logs and failure details for Submission 679 / Run ID 072e3040-e7d7-40c0-83be-e4b09b781a7a, and let me know what caused the run to finish without a valid result file, and whether there is anything we should change on our side?

Thank you again for running the competition and for your help with this issue.

Best regards,

Masahiro Kubota  
Team masapon

## Japanese Translation

AI for Industry Challenge 運営チーム御中

AI for Industry Challenge を開催し、競技期間中も参加者をサポートしてくださっていることに感謝いたします。

チーム masapon の Qualification 提出で繰り返し発生している問題について、ご連絡いたします。

直近の複数の Qualification 提出で、以下の最新 image を含めて同じ症状が発生しています。各提出 image は提出前にローカルの検証フローでは正常に通る一方で、クラウド評価環境では有効な出力が生成されないまま終了しているように見えます。ポータル上では submission は "Failed" となり、"Result file" は空で、"Stdout file" には実行ヘッダのみが出力されています。

直近の例は以下です。

- Team: masapon
- Submission ID: 679
- Run ID: 072e3040-e7d7-40c0-83be-e4b09b781a7a
- Submitted image URI:
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/masapon:submission-safe-v7-e7bba00-20260505-024538

この run では、"Stdout file" には以下のみが含まれていました。

AI for Industry Challenge - 2026
------
Date: 2026-05-04T18:00:18.474058+00:00
Team: N/A
Submission: 679
Run ID: 072e3040-e7d7-40c0-83be-e4b09b781a7a
------

また、"Result file" は空でした。

各提出前には、対応する submission image を標準的な Docker ベースの手順でローカル検証しています。

docker compose \
  -f docker/docker-compose.yaml \
  -f docker/docker-compose.submission_safe_v7.yaml \
  up

上記コマンドのログを、"local_verification.log" として添付します。

ローカル検証で使った compose file は以下です。
https://github.com/masahiro-kubota/aic/blob/5-4/docker/docker-compose.yaml
https://github.com/masahiro-kubota/aic/blob/5-4/docker/docker-compose.submission_safe_v7.yaml
この override file は、model サービスの docker image 名を上書きしているだけです。

Submission 679 / Run ID 072e3040-e7d7-40c0-83be-e4b09b781a7a に対応するクラスタ側のログや失敗詳細を確認いただき、この run が有効な result file を生成しないまま終了した原因と、こちらで修正すべき点があればご教示いただけないでしょうか。

改めまして、本大会の開催と本件へのご対応に感謝いたします。

どうぞよろしくお願いいたします。

Masahiro Kubota  
Team masapon

## Notes

- Portal observation from multiple recent attempts:
  - status: `Failed`
  - execution time: roughly `103` to `128` seconds
  - submitted file: present
  - result file: empty
  - stdout file: header only
- This draft is based on the current submission image that was pushed and confirmed in ECR.
