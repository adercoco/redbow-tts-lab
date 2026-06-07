import SwiftUI
#if canImport(FoundationModels)
import FoundationModels
#endif

struct ContentView: View {
    @State private var draft = ""
    @State private var mode: ReplyMode = .loved
    @State private var posts: [Post]
    @State private var profile: UserProfile

    init() {
        _posts = State(initialValue: DemoStore.loadPosts())
        _profile = State(initialValue: DemoStore.loadProfile())
    }

    var body: some View {
        Group {
            if profile.hasOnboarded {
                TabView {
                    FeedView(draft: $draft, mode: $mode, posts: $posts, profile: $profile)
                        .tabItem {
                            Image(systemName: "house.fill")
                            Text("Home")
                        }

                    ProfileView(posts: posts, profile: $profile)
                        .tabItem {
                            Image(systemName: "person.crop.circle")
                            Text("Profile")
                        }
                }
            } else {
                OnboardingView(profile: $profile, mode: $mode)
            }
        }
        .tint(.black)
        .onChange(of: profile) { _, newValue in
            DemoStore.save(profile: newValue, posts: posts)
        }
        .onChange(of: posts) { _, newValue in
            DemoStore.save(profile: profile, posts: newValue)
        }
    }
}

private struct OnboardingView: View {
    @Binding var profile: UserProfile
    @Binding var mode: ReplyMode

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("threads")
                            .font(.system(size: 34, weight: .bold))
                        Text("歡迎來睡 thread")
                            .font(.system(size: 18, weight: .semibold))
                    }
                    .padding(.top, 28)

                    Text("真正的 thread 文常常沒人留言。在這裡就把它當私密日記寫，讓假網友慢慢給你愛心、留言，取暖一下。")
                        .font(.system(size: 16))
                        .lineSpacing(4)
                        .foregroundStyle(.secondary)

                    VStack(alignment: .leading, spacing: 10) {
                        Text("你的綽號")
                            .font(.system(size: 14, weight: .bold))
                        TextField("例如：睡、棉被、今天好累", text: $profile.displayName)
                            .font(.system(size: 18, weight: .semibold))
                            .padding(14)
                            .background(Color(.secondarySystemBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }

                    VStack(alignment: .leading, spacing: 14) {
                        Toggle(isOn: $profile.usesAvatar) {
                            VStack(alignment: .leading, spacing: 3) {
                                Text("使用大頭貼")
                                    .font(.system(size: 15, weight: .bold))
                                Text("可以先不用，之後在 Profile 再改。")
                                    .font(.system(size: 13))
                                    .foregroundStyle(.secondary)
                            }
                        }

                        if profile.usesAvatar {
                            HStack(spacing: 14) {
                                MeAvatar(profile: profile, size: 62)
                                TextField("大頭貼文字", text: $profile.avatarText)
                                    .font(.system(size: 16, weight: .semibold))
                                    .padding(12)
                                    .background(Color(.secondarySystemBackground))
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("每篇文都可以選模式")
                            .font(.system(size: 16, weight: .bold))
                        Text("模式會決定愛心和留言長出來的速度、數量和語氣。")
                            .font(.system(size: 14))
                            .foregroundStyle(.secondary)
                        ModeSelector(mode: $mode)
                    }

                    Button {
                        profile.finishOnboarding()
                    } label: {
                        Text("下一步")
                            .font(.system(size: 17, weight: .bold))
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity, minHeight: 50)
                            .background(profile.displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.gray.opacity(0.5) : Color.black)
                            .clipShape(Capsule())
                    }
                    .disabled(profile.displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .padding(.top, 4)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 28)
            }
            .background(Color(.systemBackground))
        }
    }
}

private struct FeedView: View {
    @Binding var draft: String
    @Binding var mode: ReplyMode
    @Binding var posts: [Post]
    @Binding var profile: UserProfile
    @State private var localModelStatus = LocalReplyGenerator.statusText

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(spacing: 0) {
                    ThreadComposer(draft: $draft, mode: $mode, profile: profile, localModelStatus: localModelStatus, onPost: publish)

                    if posts.isEmpty {
                        EmptyFeedView()
                    } else {
                        ForEach($posts) { $post in
                            UserPostRow(post: $post, profile: profile, showsReplies: true, onReply: { replyID in
                                appendUserReply(to: post.id, parentReplyID: replyID)
                            })
                        }
                    }
                }
            }
            .background(Color(.systemBackground))
            .navigationTitle("threads")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 12, weight: .bold))
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Button(action: publish) {
                        Text("Post")
                            .font(.system(size: 15, weight: .bold))
                    }
                    .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
    }

    private func publish() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        let activeMode = mode == .random ? ReplyMode.randomConcrete() : mode
        let newPost = Post(
            body: text,
            time: "now",
            mode: activeMode,
            targetLikes: activeMode.likeCount,
            targetReposts: activeMode.repostCount,
            targetReplies: []
        )
        let postID = newPost.id

        withAnimation(.spring(response: 0.34, dampingFraction: 0.88)) {
            posts.insert(newPost, at: 0)
            draft = ""
        }

        Task {
            let result = await LocalReplyGenerator.generateReplies(for: text, mode: activeMode)
            await MainActor.run {
                localModelStatus = result.status
                guard let index = posts.firstIndex(where: { $0.id == postID }) else { return }
                posts[index].targetReplies = result.replies
            }
            await animateEngagement(for: postID, mode: activeMode)
        }
    }

    @MainActor
    private func appendUserReply(to postID: UUID, parentReplyID: UUID?) {
        guard let postIndex = posts.firstIndex(where: { $0.id == postID }) else { return }
        let reply = Reply(
            author: nil,
            text: parentReplyID == nil ? "我想回一下這篇。" : "回你這句，我懂。",
            time: "now",
            likes: 0,
            nestedReplies: []
        )

        withAnimation(.spring(response: 0.34, dampingFraction: 0.86)) {
            if let parentReplyID, let replyIndex = posts[postIndex].replies.firstIndex(where: { $0.id == parentReplyID }) {
                posts[postIndex].replies[replyIndex].nestedReplies.append(reply)
            } else {
                posts[postIndex].replies.append(reply)
            }
        }
    }

    private func animateEngagement(for postID: UUID, mode: ReplyMode) async {
        let plan = mode.engagementPlan

        for burst in plan.likeBursts {
            try? await Task.sleep(nanoseconds: UInt64(plan.likeInterval * 1_000_000_000))
            await MainActor.run {
                guard let index = posts.firstIndex(where: { $0.id == postID }) else { return }
                withAnimation(.snappy(duration: 0.22)) {
                    posts[index].likes = min(posts[index].targetLikes, posts[index].likes + burst)
                    posts[index].reposts = min(posts[index].targetReposts, posts[index].reposts + max(0, burst / plan.repostDivisor))
                }
            }
        }

        let generatedReplies = await MainActor.run {
            posts.first(where: { $0.id == postID })?.targetReplies ?? mode.replies
        }

        for reply in generatedReplies {
            try? await Task.sleep(nanoseconds: UInt64(plan.replyInterval * 1_000_000_000))
            await MainActor.run {
                guard let index = posts.firstIndex(where: { $0.id == postID }) else { return }
                withAnimation(.spring(response: 0.36, dampingFraction: 0.86)) {
                    posts[index].replies.append(reply)
                    posts[index].likes = min(posts[index].targetLikes, posts[index].likes + plan.replyLikeBump)
                }
            }
        }
    }
}

private struct ThreadComposer: View {
    @Binding var draft: String
    @Binding var mode: ReplyMode
    let profile: UserProfile
    let localModelStatus: String
    let onPost: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top, spacing: 12) {
                MeAvatar(profile: profile, size: 42)

                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 5) {
                        Text(profile.shownName)
                            .font(.system(size: 15, weight: .bold))
                        Text("@\(profile.shownHandle)")
                            .font(.system(size: 14))
                            .foregroundStyle(.secondary)
                    }

                    ZStack(alignment: .topLeading) {
                        if draft.isEmpty {
                            Text("有什麼新鮮事？")
                                .font(.system(size: 18))
                                .foregroundStyle(.secondary)
                                .padding(.top, 8)
                                .padding(.leading, 4)
                        }

                        TextEditor(text: $draft)
                            .font(.system(size: 18))
                            .frame(minHeight: 72, maxHeight: 128)
                            .scrollContentBackground(.hidden)
                            .padding(.horizontal, -5)
                            .padding(.top, -8)
                    }

                    ModeSelector(mode: $mode)

                    HStack(spacing: 6) {
                        Image(systemName: "cpu")
                            .font(.system(size: 12, weight: .bold))
                        Text(localModelStatus)
                            .font(.system(size: 12, weight: .semibold))
                            .lineLimit(1)
                    }
                    .foregroundStyle(.secondary)

                    HStack {
                        Text(mode.subtitle)
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)

                        Spacer()

                        Button(action: onPost) {
                            Text("Post")
                                .font(.system(size: 15, weight: .bold))
                                .foregroundStyle(.white)
                                .frame(width: 72, height: 36)
                                .background(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.gray.opacity(0.45) : Color.black)
                                .clipShape(Capsule())
                        }
                        .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 16)

            Divider()
        }
    }
}

private struct EmptyFeedView: View {
    var body: some View {
        VStack(spacing: 9) {
            Text("還沒有貼文")
                .font(.system(size: 16, weight: .semibold))
            Text("你的文章會一路記錄在這裡。")
                .font(.system(size: 14))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 72)
        .padding(.horizontal, 28)
    }
}

private struct ModeSelector: View {
    @Binding var mode: ReplyMode
    private let columns = [
        GridItem(.flexible(), spacing: 8),
        GridItem(.flexible(), spacing: 8),
        GridItem(.flexible(), spacing: 8)
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("選擇回饋模式")
                    .font(.system(size: 14, weight: .bold))
                Spacer()
                Text(mode.rawValue)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(columns: columns, spacing: 8) {
                ForEach(ReplyMode.allCases) { item in
                    Button {
                        mode = item
                    } label: {
                        VStack(alignment: .leading, spacing: 7) {
                            HStack {
                                Image(systemName: item.icon)
                                    .font(.system(size: 15, weight: .bold))
                                Spacer()
                                if mode == item {
                                    Image(systemName: "checkmark.circle.fill")
                                        .font(.system(size: 15, weight: .bold))
                                }
                            }

                            Text(item.rawValue)
                                .font(.system(size: 14, weight: .bold))
                                .lineLimit(1)
                                .minimumScaleFactor(0.76)

                            Text(item.shortDescription)
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(mode == item ? .white.opacity(0.72) : .secondary)
                                .lineLimit(1)
                        }
                        .foregroundStyle(mode == item ? .white : .primary)
                        .frame(maxWidth: .infinity, minHeight: 76, alignment: .leading)
                        .padding(10)
                        .background(mode == item ? Color.black : Color(.secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

private struct ProfileView: View {
    let posts: [Post]
    @Binding var profile: UserProfile
    @State private var isEditingAvatar = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 0) {
                    profileHeader

                    if posts.isEmpty {
                        VStack(spacing: 8) {
                            Text("還沒有貼文")
                                .font(.system(size: 16, weight: .semibold))
                            Text("回首頁寫第一篇。")
                                .font(.system(size: 14))
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.top, 44)
                    } else {
                        ForEach(posts) { post in
                            UserPostRow(post: .constant(post), profile: profile, showsReplies: false, onReply: { _ in })
                        }
                    }
                }
            }
            .background(Color(.systemBackground))
            .navigationTitle("Profile")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $isEditingAvatar) {
                AvatarEditor(profile: $profile)
                    .presentationDetents([.medium])
            }
        }
    }

    private var profileHeader: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    TextField("名字", text: $profile.displayName)
                        .font(.system(size: 28, weight: .bold))
                    HStack(spacing: 0) {
                        Text("@")
                            .font(.system(size: 15))
                            .foregroundStyle(.secondary)
                    TextField("handle", text: $profile.handle)
                            .font(.system(size: 15))
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .foregroundStyle(.secondary)
                    }
                    Text("日記 / 簡介")
                        .font(.system(size: 15))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    isEditingAvatar = true
                } label: {
                    MeAvatar(profile: profile, size: 72)
                        .overlay(alignment: .bottomTrailing) {
                            Image(systemName: "pencil.circle.fill")
                                .font(.system(size: 22, weight: .bold))
                                .symbolRenderingMode(.palette)
                                .foregroundStyle(.white, .black)
                                .background(Circle().fill(.white))
                        }
                }
                .buttonStyle(.plain)
            }

            ZStack(alignment: .topLeading) {
                if profile.diary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text("寫一點給自己看的日記或簡介...")
                        .font(.system(size: 15))
                        .foregroundStyle(.secondary)
                        .padding(.top, 8)
                        .padding(.leading, 5)
                }

                TextEditor(text: $profile.diary)
                    .font(.system(size: 15))
                    .frame(minHeight: 78)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, -5)
                    .padding(.top, -8)
            }

            HStack(spacing: 4) {
                Text("\(posts.count)")
                    .font(.system(size: 14, weight: .bold))
                Text("posts")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 22) {
                Text("Posts")
                    .font(.system(size: 15, weight: .bold))
                    .frame(maxWidth: .infinity)
            }
            .padding(.top, 8)
        }
        .padding(.horizontal, 16)
        .padding(.top, 18)
        .padding(.bottom, 14)
        .overlay(alignment: .bottom) {
            Divider()
        }
    }
}

private struct UserPostRow: View {
    @Binding var post: Post
    let profile: UserProfile
    let showsReplies: Bool
    let onReply: (UUID?) -> Void
    @State private var isExpanded = true

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top, spacing: 12) {
                VStack(spacing: 0) {
                    MeAvatar(profile: profile, size: 42)

                    if showsReplies && !post.replies.isEmpty {
                        Rectangle()
                            .fill(Color(.separator).opacity(0.56))
                            .frame(width: 2)
                            .padding(.top, 8)
                    }
                }

                VStack(alignment: .leading, spacing: 9) {
                    HStack(spacing: 5) {
                        Text(profile.shownName)
                            .font(.system(size: 15, weight: .bold))
                        Text("@\(profile.shownHandle)")
                            .foregroundStyle(.secondary)
                        Text("· \(post.time)")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Image(systemName: "ellipsis")
                            .foregroundStyle(.secondary)
                    }
                    .font(.system(size: 14))

                    Text(post.body)
                        .font(.system(size: 16.5))
                        .lineSpacing(3)
                        .fixedSize(horizontal: false, vertical: true)

                    HStack(spacing: 18) {
                        Button {
                            postToggleLike()
                        } label: {
                            Metric(icon: post.isLiked ? "heart.fill" : "heart", value: "\(post.visibleLikes)")
                                .foregroundStyle(post.isLiked ? .red : .secondary)
                        }
                        .buttonStyle(.plain)

                        Button {
                            onReply(nil)
                        } label: {
                            Metric(icon: "bubble.right", value: "\(post.totalReplyCount)")
                        }
                        .buttonStyle(.plain)

                        Metric(icon: "arrow.2.squarepath", value: "\(post.reposts)")
                        Metric(icon: "paperplane", value: "")
                    }
                    .padding(.top, 2)
                    .contentTransition(.numericText())

                    if showsReplies && !post.replies.isEmpty {
                        Button {
                            withAnimation(.snappy(duration: 0.22)) { isExpanded.toggle() }
                        } label: {
                            Text(isExpanded ? "隱藏回覆" : "查看 \(post.totalReplyCount) 則回覆")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .padding(.top, 2)

                        if isExpanded {
                            ForEach(post.replies.prefix(4)) { reply in
                            ReplyRow(reply: reply, onLike: {}, onReply: { onReply(reply.id) })
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)

            Divider()
                .padding(.leading, 70)
        }
    }

    private func postToggleLike() {
        post.isLiked.toggle()
    }
}

private struct ReplyRow: View {
    let reply: Reply
    let onLike: () -> Void
    let onReply: () -> Void
    @State private var isLiked = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                if let author = reply.author {
                    Avatar(person: author, size: 32)
                } else {
                    Circle()
                        .fill(Color(.secondarySystemBackground))
                        .frame(width: 32, height: 32)
                        .overlay(Image(systemName: "person.fill").font(.system(size: 13)).foregroundStyle(.secondary))
                }

                VStack(alignment: .leading, spacing: 5) {
                    HStack(spacing: 4) {
                        Text(reply.displayName)
                            .font(.system(size: 14, weight: .bold))
                        Text("@\(reply.handle) · \(reply.time)")
                            .font(.system(size: 13))
                            .foregroundStyle(.secondary)
                    }

                    Text(reply.text)
                        .font(.system(size: 15))
                        .lineSpacing(2)

                    HStack(spacing: 14) {
                        Button {
                            isLiked.toggle()
                            onLike()
                        } label: {
                            Metric(icon: isLiked ? "heart.fill" : "heart", value: "\(reply.likes + (isLiked ? 1 : 0))")
                                .foregroundStyle(isLiked ? .red : .secondary)
                        }
                        .buttonStyle(.plain)

                        Button(action: onReply) {
                            Text("回覆")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            ForEach(reply.nestedReplies) { nested in
                HStack(alignment: .top, spacing: 10) {
                    Rectangle()
                        .fill(Color(.separator).opacity(0.5))
                        .frame(width: 2)
                        .padding(.leading, 16)
                    Circle()
                        .fill(Color(.secondarySystemBackground))
                        .frame(width: 28, height: 28)
                        .overlay(Image(systemName: "person.fill").font(.system(size: 12)).foregroundStyle(.secondary))
                    VStack(alignment: .leading, spacing: 3) {
                        Text("\(nested.displayName)  @\(nested.handle) · \(nested.time)")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Text(nested.text)
                            .font(.system(size: 14))
                    }
                }
                .padding(.leading, 32)
            }
        }
        .padding(.vertical, 9)
    }
}

private enum ReplyMode: String, CaseIterable, Identifiable, Codable {
    case loved = "被愛爆"
    case viral = "爆紅"
    case circle = "小圈圈"
    case resonance = "共鳴"
    case random = "隨機"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .loved: "heart.fill"
        case .viral: "flame.fill"
        case .circle: "person.3.fill"
        case .resonance: "waveform.path.ecg"
        case .random: "shuffle"
        }
    }

    var subtitle: String {
        switch self {
        case .loved: "很多愛心和溫柔留言"
        case .viral: "像突然爆紅一樣熱鬧"
        case .circle: "像固定朋友群回你"
        case .resonance: "很多人說你懂我"
        case .random: "讓今天的宇宙決定"
        }
    }

    var shortDescription: String {
        switch self {
        case .loved: "溫柔稱讚"
        case .viral: "快速爆量"
        case .circle: "熟人陪聊"
        case .resonance: "被懂共鳴"
        case .random: "交給宇宙"
        }
    }

    var likeCount: Int {
        switch self {
        case .loved: 728
        case .viral: 1842
        case .circle: 126
        case .resonance: 514
        case .random: 420
        }
    }

    var repostCount: Int {
        switch self {
        case .loved: 32
        case .viral: 126
        case .circle: 4
        case .resonance: 18
        case .random: 12
        }
    }

    var replies: [Reply] {
        switch self {
        case .loved: Reply.loved
        case .viral: Reply.viral + Array(Reply.loved.prefix(2))
        case .circle: Reply.circle
        case .resonance: Reply.resonance + Array(Reply.circle.prefix(1))
        case .random: Self.randomConcrete().replies
        }
    }

    static func randomConcrete() -> ReplyMode {
        [ReplyMode.loved, ReplyMode.viral, ReplyMode.circle, ReplyMode.resonance].randomElement() ?? .loved
    }

    var engagementPlan: EngagementPlan {
        switch self {
        case .loved:
            EngagementPlan(likeBursts: [8, 17, 31, 48, 84, 121, 170, 249], likeInterval: 0.20, replyInterval: 0.62, replyLikeBump: 18, repostDivisor: 24)
        case .viral:
            EngagementPlan(likeBursts: [34, 68, 116, 190, 260, 330, 410, 434], likeInterval: 0.13, replyInterval: 0.36, replyLikeBump: 36, repostDivisor: 14)
        case .circle:
            EngagementPlan(likeBursts: [4, 7, 11, 18, 25, 31, 30], likeInterval: 0.34, replyInterval: 0.85, replyLikeBump: 6, repostDivisor: 40)
        case .resonance:
            EngagementPlan(likeBursts: [12, 24, 39, 57, 80, 112, 190], likeInterval: 0.24, replyInterval: 0.58, replyLikeBump: 16, repostDivisor: 28)
        case .random:
            ReplyMode.randomConcrete().engagementPlan
        }
    }
}

private struct Post: Identifiable, Equatable, @unchecked Sendable {
    let id: UUID
    let body: String
    let time: String
    let mode: ReplyMode
    let targetLikes: Int
    let targetReposts: Int
    var targetReplies: [Reply]
    var likes = 0
    var isLiked = false
    var reposts = 0
    var replies: [Reply] = []

    init(
        id: UUID = UUID(),
        body: String,
        time: String,
        mode: ReplyMode,
        targetLikes: Int,
        targetReposts: Int,
        targetReplies: [Reply],
        likes: Int = 0,
        isLiked: Bool = false,
        reposts: Int = 0,
        replies: [Reply] = []
    ) {
        self.id = id
        self.body = body
        self.time = time
        self.mode = mode
        self.targetLikes = targetLikes
        self.targetReposts = targetReposts
        self.targetReplies = targetReplies
        self.likes = likes
        self.isLiked = isLiked
        self.reposts = reposts
        self.replies = replies
    }

    var visibleLikes: Int {
        likes + (isLiked ? 1 : 0)
    }

    var totalReplyCount: Int {
        replies.reduce(replies.count) { $0 + $1.nestedReplies.count }
    }
}

private struct EngagementPlan {
    let likeBursts: [Int]
    let likeInterval: Double
    let replyInterval: Double
    let replyLikeBump: Int
    let repostDivisor: Int
}

private struct LocalReplyResult {
    let replies: [Reply]
    let status: String
}

private enum LocalReplyGenerator {
    static var statusText: String {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, *) {
            switch SystemLanguageModel.default.availability {
            case .available:
                return "本機小模型可用"
            case .unavailable(.deviceNotEligible):
                return "本機小模型：裝置不支援"
            case .unavailable(.appleIntelligenceNotEnabled):
                return "本機小模型：Apple Intelligence 未開"
            case .unavailable(.modelNotReady):
                return "本機小模型：模型尚未準備好"
            case .unavailable:
                return "本機小模型暫不可用"
            @unknown default:
                return "本機小模型狀態未知"
            }
        }
        #endif
        return "本機小模型需要 iOS 26"
    }

    static func generateReplies(for post: String, mode: ReplyMode) async -> LocalReplyResult {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, *), case .available = SystemLanguageModel.default.availability {
            let start = ContinuousClock.now
            let instructions = """
            你是私密日記 app 裡的假 Threads 網友生成器。
            請用繁體中文，生成 3 到 5 則很像真人的短留言。
            留言要有不同個性：溫柔、共鳴、吐槽、產品感、短句。
            不要說自己是 AI。不要過度雞湯。每則 8 到 32 字。
            """
            let session = LanguageModelSession(instructions: instructions)
            let prompt = """
            貼文：\(post)
            模式：\(mode.rawValue)，\(mode.subtitle)

            請只輸出留言，每行一則，不要編號。
            """

            do {
                let response = try await session.respond(to: prompt)
                let elapsed = start.duration(to: .now)
                let lines = response.content
                    .split(separator: "\n")
                    .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                    .prefix(5)
                let replies = lines.enumerated().map { index, line in
                    Reply(author: Person.synthetic[index % Person.synthetic.count], text: line, time: "\(index + 1)m", likes: max(12, 88 - index * 13))
                }
                let milliseconds = max(1, Int(Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0) + Int(elapsed.components.seconds) * 1000)
                return LocalReplyResult(replies: replies.isEmpty ? mode.replies : replies, status: "本機小模型 \(milliseconds)ms")
            } catch {
                return LocalReplyResult(replies: mode.replies, status: "本機小模型失敗，已用備援留言")
            }
        }
        #endif
        return LocalReplyResult(replies: mode.replies, status: statusText)
    }
}

private struct Reply: Identifiable, Equatable, @unchecked Sendable {
    let id: UUID
    let author: Person?
    let text: String
    let time: String
    let likes: Int
    var nestedReplies: [Reply]

    var displayName: String { author?.name ?? "你" }
    var handle: String { author?.handle ?? "private.reply" }

    init(id: UUID = UUID(), author: Person?, text: String, time: String, likes: Int, nestedReplies: [Reply] = []) {
        self.id = id
        self.author = author
        self.text = text
        self.time = time
        self.likes = likes
        self.nestedReplies = nestedReplies
    }

    static let loved = [
        Reply(author: .mika, text: "這個概念好準。日記不是缺功能，是缺一種有人在旁邊點頭的感覺。", time: "2m", likes: 94),
        Reply(author: .nana, text: "我會用。尤其是很累但又不想找真人聊天的晚上。", time: "4m", likes: 81),
        Reply(author: .kai, text: "重點是回覆要像真的人，不要都像罐頭稱讚。", time: "7m", likes: 63),
        Reply(author: .luna, text: "這不是假紅，是把自我記錄做得比較不孤單。", time: "11m", likes: 128)
    ]

    static let viral = [
        Reply(author: .kai, text: "這篇會爆，因為很多人都想要回饋但不想承擔社交壓力。", time: "now", likes: 204),
        Reply(author: .mika, text: "拜託做出來，這比普通 journaling app 更有黏性。", time: "1m", likes: 177),
        Reply(author: .sol, text: "我已經想像到截圖會很好看了。", time: "3m", likes: 89)
    ]

    static let circle = [
        Reply(author: .nana, text: "第一版先把這個核心做漂亮，其他可以慢慢長出來。", time: "5m", likes: 33),
        Reply(author: .ren, text: "假帳號固定出現，才會像真的朋友群。", time: "9m", likes: 29),
        Reply(author: .luna, text: "小圈圈模式感覺比爆紅更耐用。", time: "12m", likes: 41)
    ]

    static let resonance = [
        Reply(author: .luna, text: "你講出那種寫完日記還是很安靜的失落感。", time: "3m", likes: 96),
        Reply(author: .sol, text: "有些話不是需要建議，只是需要被接住。", time: "8m", likes: 84),
        Reply(author: .mika, text: "這種 app 如果語氣做對，會很有陪伴感。", time: "14m", likes: 57)
    ]
}

private struct Person: Equatable, @unchecked Sendable {
    let name: String
    let handle: String
    let colorA: Color
    let colorB: Color

    var initial: String { String(name.prefix(1)).uppercased() }

    static let mika = Person(name: "Mika", handle: "mika.notes", colorA: .pink, colorB: .purple)
    static let nana = Person(name: "Nana", handle: "nana.afterhours", colorA: .blue, colorB: .mint)
    static let kai = Person(name: "Kai", handle: "kai.product", colorA: .black, colorB: .cyan)
    static let luna = Person(name: "Luna", handle: "luna.moonlog", colorA: .purple, colorB: .indigo)
    static let ren = Person(name: "Ren", handle: "ren.builds", colorA: .orange, colorB: .red)
    static let sol = Person(name: "Sol", handle: "sol.says", colorA: .yellow, colorB: .green)
    static let synthetic: [Person] = [.mika, .nana, .kai, .luna, .ren, .sol]

    static func byHandle(_ handle: String?) -> Person? {
        guard let handle else { return nil }
        return synthetic.first { $0.handle == handle }
    }

    static func == (lhs: Person, rhs: Person) -> Bool {
        lhs.handle == rhs.handle
    }
}

private struct UserProfile: Codable, Equatable {
    var displayName = ""
    var handle = "sleep.diary"
    var diary = ""
    var avatarText = "睡"
    var palette = AvatarPalette.moon
    var usesAvatar = true
    var hasOnboarded = false

    var shownName: String {
        let trimmed = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "睡" : trimmed
    }

    var shownHandle: String {
        let cleaned = handle.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "@", with: "")
            .replacingOccurrences(of: " ", with: ".")
        return cleaned.isEmpty ? "sleep.diary" : cleaned
    }

    mutating func finishOnboarding() {
        displayName = shownName
        if avatarText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            avatarText = String(shownName.prefix(2))
        }
        handle = shownHandle
        hasOnboarded = true
    }
}

private enum AvatarPalette: String, CaseIterable, Identifiable, Codable {
    case moon = "睡眠黑"
    case candy = "甜夢粉"
    case ocean = "夜海藍"
    case moss = "棉被綠"

    var id: String { rawValue }

    var colors: [Color] {
        switch self {
        case .moon: [.black, .gray]
        case .candy: [.pink, .purple]
        case .ocean: [.blue, .cyan]
        case .moss: [.green, .mint]
        }
    }
}

private struct AvatarEditor: View {
    @Binding var profile: UserProfile
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 22) {
                HStack {
                    Spacer()
                    MeAvatar(profile: profile, size: 108)
                    Spacer()
                }
                .padding(.top, 18)

                Toggle(isOn: $profile.usesAvatar) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("顯示大頭貼")
                            .font(.system(size: 15, weight: .bold))
                        Text("關掉後會變成簡單的預設頭像。")
                            .font(.system(size: 13))
                            .foregroundStyle(.secondary)
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("大頭貼文字")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.secondary)
                    TextField("睡", text: $profile.avatarText)
                        .font(.system(size: 18, weight: .semibold))
                        .padding(12)
                        .background(Color(.secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("色系")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.secondary)
                    HStack(spacing: 10) {
                        ForEach(AvatarPalette.allCases) { palette in
                            Button {
                                profile.palette = palette
                            } label: {
                                VStack(spacing: 7) {
                                    LinearGradient(colors: palette.colors, startPoint: .topLeading, endPoint: .bottomTrailing)
                                        .frame(width: 44, height: 44)
                                        .clipShape(Circle())
                                        .overlay(Circle().stroke(profile.palette == palette ? .black : .clear, lineWidth: 3))
                                    Text(palette.rawValue)
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.75)
                                }
                                .frame(width: 72)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                Spacer()
            }
            .padding(.horizontal, 20)
            .navigationTitle("編輯大頭貼")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                        .font(.system(size: 15, weight: .bold))
                }
            }
        }
    }
}

private struct MeAvatar: View {
    let profile: UserProfile
    let size: CGFloat

    var body: some View {
        ZStack {
            if profile.usesAvatar {
                LinearGradient(colors: profile.palette.colors, startPoint: .topLeading, endPoint: .bottomTrailing)
                Text(profile.avatarText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? String(profile.shownName.prefix(2)) : String(profile.avatarText.prefix(2)))
                    .font(.system(size: size * 0.42, weight: .black))
                    .foregroundStyle(.white)
            } else {
                Color(.secondarySystemBackground)
                Image(systemName: "person.fill")
                    .font(.system(size: size * 0.42, weight: .bold))
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
    }
}

private struct Avatar: View {
    let person: Person
    let size: CGFloat

    var body: some View {
        ZStack {
            LinearGradient(colors: [person.colorA, person.colorB], startPoint: .topLeading, endPoint: .bottomTrailing)
            Text(person.initial)
                .font(.system(size: size * 0.42, weight: .black))
                .foregroundStyle(.white)
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
    }
}

private enum DemoStore {
    private static let profileKey = "sleepThread.demo.profile"
    private static let postsKey = "sleepThread.demo.posts"
    private static let decoder = JSONDecoder()
    private static let encoder = JSONEncoder()

    static func loadProfile() -> UserProfile {
        guard
            let data = UserDefaults.standard.data(forKey: profileKey),
            let profile = try? decoder.decode(UserProfile.self, from: data)
        else {
            return UserProfile()
        }
        return profile
    }

    static func loadPosts() -> [Post] {
        guard
            let data = UserDefaults.standard.data(forKey: postsKey),
            let storedPosts = try? decoder.decode([StoredPost].self, from: data)
        else {
            return []
        }
        return storedPosts.map(\.post)
    }

    static func save(profile: UserProfile, posts: [Post]) {
        if let profileData = try? encoder.encode(profile) {
            UserDefaults.standard.set(profileData, forKey: profileKey)
        }
        if let postsData = try? encoder.encode(posts.map(StoredPost.init)) {
            UserDefaults.standard.set(postsData, forKey: postsKey)
        }
    }

    private struct StoredPost: Codable {
        let id: UUID
        let body: String
        let time: String
        let mode: ReplyMode
        let targetLikes: Int
        let targetReposts: Int
        let targetReplies: [StoredReply]
        let likes: Int
        let isLiked: Bool
        let reposts: Int
        let replies: [StoredReply]

        init(_ post: Post) {
            id = post.id
            body = post.body
            time = post.time
            mode = post.mode
            targetLikes = post.targetLikes
            targetReposts = post.targetReposts
            targetReplies = post.targetReplies.map(StoredReply.init)
            likes = post.likes
            isLiked = post.isLiked
            reposts = post.reposts
            replies = post.replies.map(StoredReply.init)
        }

        var post: Post {
            Post(
                id: id,
                body: body,
                time: time,
                mode: mode,
                targetLikes: targetLikes,
                targetReposts: targetReposts,
                targetReplies: targetReplies.map(\.reply),
                likes: likes,
                isLiked: isLiked,
                reposts: reposts,
                replies: replies.map(\.reply)
            )
        }
    }

    private struct StoredReply: Codable {
        let id: UUID
        let authorHandle: String?
        let text: String
        let time: String
        let likes: Int
        let nestedReplies: [StoredReply]

        init(_ reply: Reply) {
            id = reply.id
            authorHandle = reply.author?.handle
            text = reply.text
            time = reply.time
            likes = reply.likes
            nestedReplies = reply.nestedReplies.map(StoredReply.init)
        }

        var reply: Reply {
            Reply(
                id: id,
                author: Person.byHandle(authorHandle),
                text: text,
                time: time,
                likes: likes,
                nestedReplies: nestedReplies.map(\.reply)
            )
        }
    }
}

private struct Metric: View {
    let icon: String
    let value: String

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
            if !value.isEmpty {
                Text(value)
            }
        }
        .font(.system(size: 13, weight: .semibold))
        .foregroundStyle(.secondary)
    }
}
